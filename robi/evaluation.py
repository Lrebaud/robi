from robi.evaluation_npy import *
from robi.evaluation_torch import *
from tqdm import tqdm

def compute_pvals(scores, scores_random, device, use_torch):
    if use_torch:
        return compute_pvals_torch(scores, scores_random, device)
    return compute_pvals_npy(scores, scores_random)


def compute_univariate_score(df, random_cols, targets, device, use_torch):
    if use_torch:
        df_arr = torch.as_tensor(df[random_cols].values, device=device, dtype=torch.float)
        return compute_univariate_score_torch(df, random_cols, targets, df_arr)
    df_arr = df[random_cols].values
    return compute_univariate_score_npy(df, random_cols, targets)


def score_of_random(df, targets, device, n_random, use_torch):
    df_random = pd.DataFrame(data=np.random.uniform(0,1, (df.shape[0], int(n_random))), index=df.index)
    random_cols = df_random.columns
    df_random = pd.concat([df[sum([list(targets[x]) for x in targets], [])], df_random], axis=1)
    random_scores = compute_univariate_score(df_random, random_cols, targets, device, use_torch)
    for target in random_scores:
        random_scores[target] = abs(random_scores[target]-0.5)
    return random_scores

def univariate_evaluation(df, candidates, targets, device, scores_random, use_torch):
    scores = compute_univariate_score(df, candidates, targets, device, use_torch)
    pvals = compute_pvals(scores, scores_random, device, use_torch)
    return scores, pvals


def get_permuted_scores(df, candidates, scores_random_by_target, targets, device, n_permut, verbose, use_torch):
    cols_to_permut = sum([list(targets[x]) for x in targets], [])
    all_pscores = []

    if use_torch:
        df_arr = torch.as_tensor(df[candidates].values, device=device, dtype=torch.float)
    else:
        df_arr = df[candidates].values

    for _ in tqdm(range(n_permut),
                   desc='Computing scores of permutations',
                   bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                   disable=not verbose):
        dfp = df.copy()
        dfp.loc[dfp.index, cols_to_permut] = dfp.loc[np.random.permutation(dfp.index), cols_to_permut].values

        if use_torch:
            scores = compute_univariate_score_torch(dfp, candidates, targets, df_arr)
        else:
            scores = compute_univariate_score_npy(dfp, candidates, targets)

        pscores = compute_pvals(scores, scores_random_by_target, device, use_torch)
        all_pscores.append(pscores)
    return all_pscores
