import numpy as np


def get_comparable_pairs_npy(time, event):
    y = time
    e = event

    grid_x, grid_y = np.meshgrid(y, y)
    grid_x = np.tril(grid_x)
    grid_y = np.tril(grid_y)
    diff_truth = grid_x - grid_y

    grid_ex, grid_ey = np.meshgrid(e, e)
    valid_pairs = ((diff_truth < 0) & (grid_ex == 1))
    res1 = np.stack(np.where(valid_pairs)).T

    diff_truth = grid_y - grid_x
    valid_pairs = ((diff_truth < 0) & (grid_ey == 1))
    res2 = np.flip(np.stack(np.where(valid_pairs)).T, axis=1)

    pairs = np.concatenate([res1, res2])
    return pairs


def cindex_by_pair_npy(v_high, v_low):
    eval_comparable = (v_high < v_low).astype('float32')
    eval_non_comparable = (v_high == v_low).astype('float32')
    return eval_comparable + (eval_non_comparable * 0.5)


def compute_univariate_score_of_npy(pairs, biom_values):
    biom_values_low = biom_values[pairs[:, 0]]
    biom_values_high = biom_values[pairs[:, 1]]

    # compute feature's signs
    features_cindex_by_pair = cindex_by_pair_npy(biom_values_high, biom_values_low)
    features_cindex = 1. - features_cindex_by_pair.mean(axis=0)

    return features_cindex


def compute_univariate_score_npy(df, candidates, target_columns, biom_values):
    candidates_id = np.arange(len(candidates))

    max_chunk_size = 1e4
    if len(candidates_id) > max_chunk_size:
        n_chunks = len(candidates_id) / max_chunk_size
        chunks = np.array_split(candidates_id, n_chunks)
    else:
        chunks = [candidates_id]

    all_scores_by_target = {}
    for target in target_columns:
        # get comparable train pairs values
        pairs = get_comparable_pairs_npy(df[target_columns[target][0]].values,
                                         df[target_columns[target][1]].astype('bool').values)
        univ_scores = []
        for chunk in chunks:
            univ_scores.append(compute_univariate_score_of_npy(pairs, biom_values[:, chunk]))
        all_scores_by_target[target] = np.concatenate(univ_scores)

    return all_scores_by_target


def compute_pvals_target_npy(scores, random_scores):
    scores = abs(scores - 0.5)
    n_random = len(random_scores)

    max_chunk_size = 1e2

    if len(scores) > max_chunk_size:
        n_chunks = int(len(scores) / max_chunk_size)
        chunks = np.array_split(scores, n_chunks)
    else:
        chunks = [scores]

    all_pvals = []
    for chunk_values in chunks:
        chunk_values = chunk_values[:, None]
        pvals = ((random_scores >= chunk_values).sum(axis=1) + 1) / (n_random + 1)
        all_pvals.append(pvals)
    all_pvals = np.concatenate(all_pvals)
    return all_pvals


def compute_pvals_npy(scores, random_scores):
    pvals = {}
    for target in scores:
        pvals[target] = compute_pvals_target_npy(scores[target], random_scores[target])
    return pvals
