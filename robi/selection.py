import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import ray
from multipy.fdr import tst
from tqdm import tqdm
import seaborn as sns
from robi.evaluation import univariate_evaluation, score_of_random
from robi.preselection import primary_selection
from robi.utils import *

sns.set_theme(style="whitegrid")



def get_permuted_scores(df, candidates, scores_random, targets, device, n_permut, verbose):
    cols_to_permut = sum([list(targets[x]) for x in targets], [])
    all_pscores = []
    for _ in tqdm(range(n_permut),
                   desc='Computing scores of permutations',
                   bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                   disable=not verbose):
        dfp = df.copy()
        dfp.loc[dfp.index, cols_to_permut] = dfp.loc[np.random.permutation(dfp.index), cols_to_permut].values
        pscores = univariate_evaluation(dfp, candidates, targets, device, scores_random)
        all_pscores.append(pscores)
    return all_pscores


def get_best_candidates_per_cluster(scores, clusters):
    representative = []
    for cluster in clusters:
        if len(cluster) > 1:
            max_pvals = scores.loc[cluster]
            best_idx = np.argmin(max_pvals.values)
            representative.append(max_pvals.index[best_idx])
        else:
            representative.append(cluster[0])
    return np.array(representative)


def get_sel_by_permissiveness(scores, corr_clusters, targets):
    all_n_sel = []

    for target in targets:
        mask_target = scores['target'] == target
        pvals_target = scores.loc[mask_target, 'p_value']

        candidates = get_best_candidates_per_cluster(pvals_target, corr_clusters)
        pvals_target = pvals_target.loc[candidates].values

        for permissiveness in np.arange(0.01, 1.01, 0.01).round(2):
            selected = candidates[tst(pvals_target, q=permissiveness)]
            all_n_sel.append({
                'permissiveness': permissiveness,
                'target': target,
                'n_selected': len(selected),
                'selected': selected,
            })
    return pd.DataFrame(all_n_sel).set_index(['target', 'permissiveness'])

@ray.remote
def get_nfp_by_permissiveness_worker(pscores, chunk, corr_clusters, targets):
    all_res = []
    for x in chunk:
        s = pscores[x]
        sel = get_sel_by_permissiveness(s, corr_clusters, targets)
        sel = sel.reset_index()
        sel = sel.drop(columns=['selected'])
        sel = sel.rename(columns={'n_selected': 'n_FP'})
        all_res.append(sel)
    return all_res


def get_nfp_by_permissiveness(pscores, corr_clusters, targets, n_workers):
    n_permut = len(pscores)
    chunks = np.array_split(np.arange(n_permut), n_workers)
    pscores_id = ray.put(pscores)
    corr_clusters_id = ray.put(corr_clusters)
    workers = [get_nfp_by_permissiveness_worker.remote(pscores_id, x,
                                                       corr_clusters_id, targets) for x in chunks]
    res = sum(ray.get(workers), [])
    res = pd.concat(res).reset_index(drop=True)
    res = res.set_index(['target', 'permissiveness'])
    return res


def get_permissiveness_effect(scores, pscores, corr_clusters, targets, n_workers, verbose):
    list_targets = list(targets.keys())
    sel_by_permissiveness = get_sel_by_permissiveness(scores, corr_clusters, list_targets)
    nfp_by_permissiveness = get_nfp_by_permissiveness(pscores, corr_clusters, list_targets, n_workers)
    return sel_by_permissiveness, nfp_by_permissiveness

def plot_permissiveness_effect(sel_by_permissiveness, nfp_by_permissiveness, targets):

    for target in targets:

        sns.lineplot(sel_by_permissiveness.loc[target], x='permissiveness', y='n_selected',
                     label='Number of\nselected candidates', color='#ec6602')
        sns.lineplot(nfp_by_permissiveness.loc[target], x='permissiveness', y='n_FP',
                     label='Number of\nfalse positives', errorbar=('pi', 95), color='#009999')
        plt.title(target)
        plt.ylabel('')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

def make_selection(df,
                   candidates,
                   targets,
                   confounders=None,
                   known=None,
                   strata=None,
                   max_corr_cluster=0.5,
                   n_uni_pval=100,
                   n_fp_estimate=10,
                   verbose=True,
                   n_workers=1,
                   device="cpu"):
    if verbose:
        print('Selection started...')

    if torch_installed(verbose=True) and device != 'cuda':
        print(f"device is {device}. If you have a GPU, switching to cuda will provide significant speed gains.")

    if confounders is None:
        confounders = []
    if known is None:
        known = []
    if isinstance(strata, list):
        if len(strata) == 0:
            strata = None
    confounders = list(set(confounders+known))

    if isinstance(candidates, np.ndarray):
        candidates = candidates.tolist()

    df, targets = format_targets(df, targets)

    # drop cases where confounders are not defined
    if len(confounders) > 0:
        df = df[df[confounders].isna().sum(axis=1) == 0]

    df = normalize_columns(df, candidates + confounders)
    if verbose and len(confounders) > 0:
        check_confounders(df, confounders, targets, strata)

    # perform the preselection
    candidates, corr_clusters, candidates_correlations = primary_selection(df,
                                                                           candidates,
                                                                           confounders,
                                                                           strata,
                                                                           known,
                                                                           targets,
                                                                           max_corr_cluster,
                                                                           verbose,
                                                                           n_workers)

    scores_random = score_of_random(df, targets, device, n_uni_pval)
    scores = univariate_evaluation(df, candidates, targets, device, scores_random)
    if verbose:
        check_n_random(scores, targets, n_uni_pval)

    pscores = get_permuted_scores(df, candidates, scores_random, targets, device, n_fp_estimate, verbose)

    sel_by_permissiveness, nfp_by_permissiveness = get_permissiveness_effect(scores, pscores,
                                                                             corr_clusters, targets,
                                                                             n_workers, verbose)
    if verbose:
        plot_permissiveness_effect(sel_by_permissiveness, nfp_by_permissiveness, targets)


    nfp_by_permissiveness['n_selected'] = sel_by_permissiveness.loc[nfp_by_permissiveness.index, 'n_selected']
    nfp_by_permissiveness['P_only_FP'] = nfp_by_permissiveness['n_FP'] >= nfp_by_permissiveness['n_selected']

    nfp_by_permissiveness_g = nfp_by_permissiveness.groupby(['target', 'permissiveness'])
    nfp_mean = nfp_by_permissiveness_g['n_FP'].mean().round(1)
    nfp_cil = nfp_by_permissiveness_g['n_FP'].quantile(0.025).round(2)
    nfp_cih = nfp_by_permissiveness_g['n_FP'].quantile(0.975).round(2)
    nfp_mean = nfp_mean.round(1).astype('str') + ' (' + nfp_cil.astype('str') + '-' + nfp_cih.astype('str') + ')'
    p_only_FP = nfp_by_permissiveness_g['P_only_FP'].mean()

    sel_by_permissiveness = pd.concat([sel_by_permissiveness, nfp_mean, p_only_FP], axis=1)

    sel_by_permissiveness = sel_by_permissiveness[['n_selected', 'n_FP', 'P_only_FP', 'selected']]

    scores.index.name = 'candidate'
    scores = scores.reset_index()
    scores = scores.set_index(['candidate', 'target'])

    return sel_by_permissiveness, scores