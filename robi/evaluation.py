import numpy as np
import pandas as pd
from robi.evaluation_npy import *
from robi.evaluation_torch import *
from robi.utils import torch_installed

def compute_pvals(scores, scores_random, device):
    if torch_installed():
        return compute_pvals_torch(scores, scores_random, device)
    return compute_pvals_npy(scores, scores_random)


def compute_univariate_score(df, random_cols, targets, device):
    if torch_installed():
        return compute_univariate_score_torch(df, random_cols, targets, device)
    return compute_univariate_score_npy(df, random_cols, targets)


def score_of_random(df, targets, device, n_random):
    df_random = pd.DataFrame(data=np.random.uniform(0,1, (df.shape[0], int(n_random))), index=df.index)
    random_cols = df_random.columns
    df_random = pd.concat([df[sum([list(targets[x]) for x in targets], [])], df_random], axis=1)
    return compute_univariate_score(df_random, random_cols, targets, device)


def univariate_evaluation(df, candidates, targets, device, scores_random):
    scores = compute_univariate_score(df, candidates, targets, device)
    return compute_pvals(scores, scores_random, device)