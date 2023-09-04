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

def get_best_candidates_per_cluster(scores, clusters):
    representative = []
    for cluster in clusters:
        if len(cluster) > 1:
            max_pvals = scores.loc[cluster]
            best_idx = np.argmin(max_pvals.values)
            representative.append(max_pvals.index[best_idx])
        else:
            representative.append(cluster[0])
    return representative


def fdr_control_on_target(scores, clusters, permissiveness, target):
    candidates = get_best_candidates_per_cluster(scores[target + '_pval'], clusters)
    target_scores = scores.loc[candidates]
    target_scores['pass_' + target] = tst(target_scores[target + '_pval'].values, q=permissiveness)
    return target_scores


def get_permuted_scores(df, candidates, scores_random, targets, device, n_permut, verbose):
    cols_to_permut = sum([list(targets[x]) for x in targets], [])
    all_pscores = []
    for it in tqdm(range(n_permut),
                   desc='Computing scores of permutations',
                   bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                   disable=not verbose):
        dfp = df.copy()
        dfp.loc[dfp.index, cols_to_permut] = dfp.loc[np.random.permutation(dfp.index), cols_to_permut].values
        pscores = univariate_evaluation(dfp, candidates, targets, device, scores_random)
        all_pscores.append(pscores)
    return all_pscores


@ray.remote
def get_nfp_all_permissiveness_worker(pscores, chunk, corr_clusters, permissivenesses, target):
    all_res = []
    for x in chunk:
        s = pscores[x]
        sel_candidates = get_best_candidates_per_cluster(s[target + '_pval'], corr_clusters)
        target_scores = s.loc[sel_candidates]
        all_pass = {}
        for p in permissivenesses:
            all_res.append({
                'permissiveness': p,
                'n_fp_' + target: tst(target_scores[target + '_pval'].values, q=p).sum()
            })
    return all_res


def get_nfp_all_permissiveness(perm_values, target, pscores, corr_clusters, n_workers):
    n_permut = len(pscores)
    chunks = np.array_split(np.arange(n_permut), n_workers)
    pscores_id = ray.put(pscores)
    corr_clusters_id = ray.put(corr_clusters)
    workers = [get_nfp_all_permissiveness_worker.remote(pscores_id, x,
                                                        corr_clusters_id, perm_values, target) for x in chunks]
    res = sum(ray.get(workers), [])
    res = pd.DataFrame(res)
    res = res.set_index('permissiveness')
    return res


def fdr_control(scores, clusters, permissiveness, targets):
    all_target_scores = {}
    for target in targets:
        all_target_scores[target] = fdr_control_on_target(scores, clusters, permissiveness, target)
    return all_target_scores


def get_nsel_by_permissiveness(scores, corr_clusters):
    all_n_sel = []
    targets = np.unique([x.split('_pval')[0] for x in scores.columns]).tolist()
    for permissiveness in np.arange(0.01, 1.01, 0.01).round(2):
        sel = fdr_control(scores, corr_clusters, permissiveness, targets=targets)
        r = {'permissiveness': permissiveness}
        for target in targets:
            r['sel_' + target] = sel[target][sel[target]['pass_' + target]].index.tolist()
            r['n_sel_' + target] = sel[target]['pass_' + target].sum()
        all_n_sel.append(r)
    return pd.DataFrame(all_n_sel).set_index('permissiveness')


def plot_permissiveness_effect(scores, pscores, corr_clusters, targets, n_workers, verbose):
    nsel_by_permissiveness = get_nsel_by_permissiveness(scores, corr_clusters)
    plotid = 0
    if verbose:
        plt.figure(figsize=(12, 5), layout='constrained')
    nfp_all_permissiveness_by_target = {}
    for target in targets:
        if verbose:
            plotid += 1
            plt.subplot(1, len(list(targets.keys())), plotid)
        nfp_all_permissiveness = get_nfp_all_permissiveness(nsel_by_permissiveness.index, target,
                                                            pscores, corr_clusters, n_workers)
        nfp_all_permissiveness_by_target[target] = nfp_all_permissiveness
        if verbose:
            sns.lineplot(nsel_by_permissiveness, x='permissiveness', y='n_sel_' + target,
                         label='Number of\nselected candidates', color='#ec6602')
            sns.lineplot(nfp_all_permissiveness, x='permissiveness', y='n_fp_' + target,
                         label='Number of\nfalse positives', errorbar=('pi', 95), color='#009999')
            plt.title(target)
            plt.xlabel('Requested False Discovery Rate (TST\'s FDR)')
            plt.ylabel('')
            if plotid == 1:
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            else:
                plt.legend('', frameon=False)
    if verbose:
        plt.show()
    return nsel_by_permissiveness, nfp_all_permissiveness_by_target


def get_selection_results(n_fp, targets, nsel_by_permissiveness, nfp_all_permissiveness):
    all_rows = []
    for target in targets:
        row = {}
        row['target'] = target
        mean_nfp = nfp_all_permissiveness[target].groupby('permissiveness').mean()
        row['permissiveness'] = mean_nfp[mean_nfp['n_fp_' + target] < n_fp].index[-1]

        row['selection'] = nsel_by_permissiveness.loc[row['permissiveness'], 'sel_' + target]
        row['n_selected'] = nsel_by_permissiveness.loc[row['permissiveness'], 'n_sel_' + target]

        sel_fp = nfp_all_permissiveness[target].loc[row['permissiveness'], 'n_fp_' + target].values
        row['prob_only_FP'] = ((sel_fp >= row['n_selected']).sum() + 1) / (len(sel_fp) + 1)
        row['n_FP_(CI)'] = '%.2f' % np.mean(sel_fp) + ' (%.2f' % np.percentile(sel_fp, 2.5) + '-%.2f)' % np.percentile(
            sel_fp, 97.5)
        all_rows.append(row)
    all_rows = pd.DataFrame(all_rows)
    all_rows = all_rows.set_index('target')
    return all_rows.T


def make_selection(df,
                   candidates,
                   targets,
                   confounders=[],
                   strata=None,
                   confounders_to_norm=[],
                   confound_check_corr=[],
                   max_corr=0.5,
                   n_workers=1,
                   n_random=100,
                   n_permut_nfp=10,
                   verbose=True,
                   device="cpu"):
    if verbose:
        print('Selection started...')

    if torch_installed(verbose=True) and device != 'cuda':
        print(f"device is {device}. If you have a GPU, switching to cuda will provide significant speed gains.")

    # drop cases where confounders are not defined
    df = df[df[confounders].isna().sum(axis=1) == 0]

    df = normalize_columns(df, candidates + confounders_to_norm)
    if verbose:
        check_confounders(df, confounders, targets, strata)

    # perform the preselection and
    sel_candidates, corr_clusters, candidates_correlations = primary_selection(df,
                                                                               candidates,
                                                                               confounders,
                                                                               strata,
                                                                               confound_check_corr,
                                                                               targets,
                                                                               max_corr,
                                                                               verbose,
                                                                               n_workers)

    scores_random = score_of_random(df, targets, device, n_random)
    scores = univariate_evaluation(df, sel_candidates, targets, device, scores_random)
    if verbose:
        check_n_random(scores, targets, n_random)

    pscores = get_permuted_scores(df, sel_candidates, scores_random, targets, device, n_permut_nfp, verbose)

    nsel_by_permissiveness, nfp_all_permissiveness = plot_permissiveness_effect(scores, pscores,
                                                                                corr_clusters, targets,
                                                                                n_workers, verbose)

    for target in targets:
        mean_nfp = nfp_all_permissiveness[target].groupby('permissiveness').mean()
        ci_nfp_low = nfp_all_permissiveness[target].groupby('permissiveness').quantile(0.025).round(2).astype('str')
        ci_nfp_high = nfp_all_permissiveness[target].groupby('permissiveness').quantile(0.975).round(2).astype('str')
        mean_nfp = mean_nfp.round(1).astype('str') + ' (' + ci_nfp_low + '-' + ci_nfp_high + ')'
        nsel_by_permissiveness = pd.concat([nsel_by_permissiveness, mean_nfp], axis=1)

    return nsel_by_permissiveness, scores
