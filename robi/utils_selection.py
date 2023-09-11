from matplotlib import pyplot as plt
from multipy.fdr import tst
from joblib import Parallel, delayed
import seaborn as sns
import numpy as np
import pandas as pd


def get_best_candidates_per_cluster(scores, clusters):
    representative = []
    for cluster in clusters:
        if len(cluster) > 1:
            best_idx = np.argmin(scores[cluster])
            representative.append(cluster[best_idx])
        else:
            representative.append(cluster[0])
    return np.array(representative)


def get_random_candidates_per_cluster(clusters):
    representative = []
    for cluster in clusters:
        if len(cluster) > 1:
            representative.append(cluster[np.random.choice(np.arange(len(cluster)))])
        else:
            representative.append(cluster[0])
    return np.array(representative)


def get_candidates_per_cluster(scores, clusters, opti_cluster):
    if opti_cluster:
        return get_best_candidates_per_cluster(scores, clusters)
    else:
        return get_random_candidates_per_cluster(clusters)


def get_sel_by_permissiveness(pvals, corr_clusters, targets, opti_cluster):
    all_n_sel = []

    for target in targets:
        pvals_target = pvals[target]

        candidates = get_candidates_per_cluster(pvals_target, corr_clusters, opti_cluster)
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


def get_nfp_by_permissiveness(pvals_permut, corr_clusters, targets, n_jobs, opti_cluster):
    res = Parallel(n_jobs=n_jobs)(delayed(get_sel_by_permissiveness)
                                  (s, corr_clusters, targets, opti_cluster)
                                  for s in pvals_permut)
    res = pd.concat(res)
    res = res.reset_index()
    res = res.drop(columns=['selected'])
    res = res.rename(columns={'n_selected': 'n_FP'})
    res = res.set_index(['target', 'permissiveness'])
    return res


def get_permissiveness_effect(pval, pvals_permut, corr_clusters, targets, opti_cluster, n_workers):
    list_targets = list(targets.keys())
    sel_by_permissiveness = get_sel_by_permissiveness(pval, corr_clusters, list_targets, opti_cluster)
    nfp_by_permissiveness = get_nfp_by_permissiveness(pvals_permut, corr_clusters, list_targets, n_workers, opti_cluster)
    return sel_by_permissiveness, nfp_by_permissiveness


def plot_permissiveness_effect(sel_by_permissiveness, nfp_by_permissiveness, targets):
    sns.set_theme(style="whitegrid")

    for target in targets:
        sns.lineplot(sel_by_permissiveness.loc[target], x='permissiveness', y='n_selected',
                     label='Number of\nselected candidates', color='#ec6602')
        sns.lineplot(nfp_by_permissiveness.loc[target], x='permissiveness', y='n_FP',
                     label='Number of\nfalse positives', errorbar=('pi', 95), color='#009999')
        plt.title(target)
        plt.ylabel('')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()


def format_output(nfp_by_permissiveness, sel_by_permissiveness, scores, pvals, candidates, use_torch):
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

    return sel_by_permissiveness, scores
