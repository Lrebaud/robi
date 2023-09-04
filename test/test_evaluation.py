import pandas as pd
import numpy as np
from lifelines.utils import concordance_index
from robi.evaluation_torch import compute_univariate_score_torch, compute_pvals_torch
from robi.evaluation_npy import compute_univariate_score_npy, compute_pvals_npy
from robi.evaluation import score_of_random

def test_evaluation():
    df = pd.read_csv('../data/DLBCL_test.csv')
    targets = {
      'PFS': ('PFS_months', 'PFS_event_happened'),
    }

    n_fakes = 100

    df_random = pd.DataFrame(data=np.random.uniform(0,1, (df.shape[0], int(n_fakes))), index=df.index)
    df_random.columns = df_random.columns.astype('str')
    random_cols = df_random.columns
    df = pd.concat([df, df_random], axis=1)

    ref_cindex = []
    for c in random_cols:
        ref_cindex.append(concordance_index(event_times=df['PFS_months'],
                                       predicted_scores=-df[c],
                                       event_observed=df['PFS_event_happened']))
    ref_cindex = np.array(ref_cindex)

    scores_npy = compute_univariate_score_npy(df, random_cols, targets)
    assert np.allclose(ref_cindex, scores_npy['PFS'].values, atol=1e-3)

    scores_torch = compute_univariate_score_torch(df, random_cols, targets, device='cuda')
    assert np.allclose(ref_cindex, scores_torch['PFS'].values, atol=1e-3)

    scores_random = score_of_random(df, targets, 'cuda', 1000)
    pval_npy = compute_pvals_npy(scores_npy, scores_random)
    pval_torch = compute_pvals_torch(scores_npy, scores_random, 'cuda')

    assert np.allclose(pval_npy['PFS_pval'].values, pval_torch['PFS_pval'].values, atol=1e-3)
