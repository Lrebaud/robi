from matplotlib import pyplot as plt
from multipy.fdr import tst
from tqdm import tqdm
import seaborn as sns
from robi.evaluation import univariate_evaluation, score_of_random
from robi.preselection import primary_selection
from robi.utils import *
from joblib import Parallel, delayed

sns.set_theme(style="whitegrid")

import time
import robi
import torch


def get_best_candidates_per_cluster(scores, clusters):
    representative = []
    for cluster in clusters:
        if len(cluster) > 1:
            best_idx = np.argmin(scores[cluster])
            representative.append(cluster[best_idx])
        else:
            representative.append(cluster[0])
    return np.array(representative)


def get_sel_by_permissiveness(pvals, corr_clusters, targets):
    all_n_sel = []

    for target in targets:
        pvals_target = pvals[target]

        candidates = get_best_candidates_per_cluster(pvals_target, corr_clusters)
        pvals_target = pvals_target[candidates]

        for permissiveness in np.arange(0.01, 1.01, 0.01).round(2):
            selected = candidates[tst(pvals_target, q=permissiveness)]
            all_n_sel.append({
                'permissiveness': permissiveness,
                'target': target,
                'n_selected': len(selected),
                'selected': selected,
            })
    return pd.DataFrame(all_n_sel).set_index(['target', 'permissiveness'])


def get_nfp_by_permissiveness(pvals_permut, corr_clusters, targets, n_jobs):
    res = Parallel(n_jobs=n_jobs)(delayed(get_sel_by_permissiveness)
                                  (s, corr_clusters, targets)
                                  for s in pvals_permut)
    res = pd.concat(res)
    res = res.reset_index()
    res = res.drop(columns=['selected'])
    res = res.rename(columns={'n_selected': 'n_FP'})
    res = res.set_index(['target', 'permissiveness'])
    return res


def get_permissiveness_effect(pval, pvals_permut, corr_clusters, targets, n_workers):
    list_targets = list(targets.keys())
    sel_by_permissiveness = get_sel_by_permissiveness(pval, corr_clusters, list_targets)
    nfp_by_permissiveness = get_nfp_by_permissiveness(pvals_permut, corr_clusters, list_targets, n_workers)
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
                   max_corr_cluster=1.,
                   n_uni_pval=1000,
                   n_fp_estimate=1000,
                   verbose=True,
                   n_jobs=1,
                   device="cpu"):
    if verbose:
        print('Selection started...')

    use_torch = torch_installed(verbose=True)
    if use_torch and device != 'cuda':
        print(f"device is {device}. If you have a GPU, switching to cuda will provide significant speed gains.")

    if confounders is None:
        confounders = []
    if known is None:
        known = []
    if isinstance(strata, list):
        if len(strata) == 0:
            strata = None
    confounders = list(set(confounders + known))

    if isinstance(candidates, np.ndarray):
        candidates = candidates.tolist()

    df, targets = format_targets(df, targets)

    # drop cases where confounders are not defined
    if len(confounders) > 0:
        df = df[df[confounders].isna().sum(axis=1) == 0]

    if strata is None:
        to_norm = candidates + confounders
    else:
        to_norm = list(set(candidates + confounders) - set(strata))
    df = normalize_columns(df, to_norm)

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
                                                                           n_jobs)

    scores_random = score_of_random(df, targets, device, n_uni_pval, use_torch)
    scores, pvals = univariate_evaluation(df, candidates, targets, device, scores_random, use_torch)

    if verbose:
        check_n_random(pvals, n_uni_pval, use_torch)

    pvals_permut = robi.evaluation.get_permuted_scores(df, candidates, scores_random, targets, device, n_fp_estimate,
                                                       verbose, use_torch)

    sel_by_permissiveness, nfp_by_permissiveness = get_permissiveness_effect(pvals, pvals_permut,
                                                                             corr_clusters, targets,
                                                                             n_jobs)
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

    scores_col = []
    pval_col = []
    target_col = []
    cand_col = []
    for target in scores:
        s = scores[target]
        p = pvals[target]
        if use_torch:
            s = s.cpu().numpy()
            p = p.cpu().numpy()
        s = s.tolist()
        p = p.tolist()
        scores_col += s
        pval_col += p
        target_col += [target] * len(s)
        cand_col += candidates

    scores = pd.DataFrame(data={
        'candidate': cand_col,
        'target': target_col,
        'C_index': scores_col,
        'p_value': pval_col,
    })
    scores = scores.set_index(['candidate', 'target'])

    candidates = np.array(candidates)
    sel_by_permissiveness['selected'] = [candidates[x] for x in sel_by_permissiveness['selected']]

    torch.cuda.empty_cache()
    return sel_by_permissiveness, scores
