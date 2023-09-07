import numpy as np

try:
    import torch
except:
    print('Could not load PyTorch. Consider installing or fixing it for substantial speed gains. Using Numpy instead. ')


def get_comparable_pairs_torch(time, event):
    y = torch.from_numpy(time)
    e = torch.from_numpy(event)

    grid_x, grid_y = torch.meshgrid(y, y, indexing="ij")
    grid_x = grid_x.tril()
    grid_y = grid_y.tril()
    diff_truth = grid_x - grid_y

    grid_ex, grid_ey = torch.meshgrid(e, e, indexing="ij")
    valid_pairs = ((diff_truth < 0) & (grid_ex == 1))
    res1 = torch.stack(torch.where(valid_pairs)).T

    diff_truth = grid_y - grid_x
    valid_pairs = ((diff_truth < 0) & (grid_ey == 1))
    res2 = torch.stack(torch.where(valid_pairs)).T.flip(1)

    pairs = torch.cat([res1, res2]).numpy()
    return pairs


def cindex_by_pair_torch(v_high, v_low):
    eval_comparable = (v_high < v_low).float()
    eval_non_comparable = (v_high == v_low).float()
    return eval_comparable + (eval_non_comparable * 0.5)


def compute_univariate_score_of_torch(pairs, biom_values):
    biom_values_low = biom_values[pairs[:, 0]]
    biom_values_high = biom_values[pairs[:, 1]]

    # compute feature's signs
    features_cindex_by_pair = cindex_by_pair_torch(biom_values_high, biom_values_low)
    features_cindex = features_cindex_by_pair.mean(dim=0)

    return features_cindex


def compute_univariate_score_torch(df, candidates, target_columns, biom_values):
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
        pairs = get_comparable_pairs_torch(df[target_columns[target][0]].values,
                                           df[target_columns[target][1]].astype('bool').values)
        univ_scores = []
        for chunk in chunks:
            univ_scores.append(compute_univariate_score_of_torch(pairs, biom_values[:, chunk]))
        all_scores_by_target[target] = torch.cat(univ_scores)

    return all_scores_by_target


def compute_pvals_target_torch(scores, random_scores):
    scores = abs(scores - 0.5)
    n_random = len(random_scores)

    max_chunk_size = 1e2

    if len(scores) > max_chunk_size:
        n_chunks = int(len(scores) / max_chunk_size)
        chunks = torch.split(scores, n_chunks)
    else:
        chunks = [scores]

    all_pvals = []
    for chunk_values in chunks:
        chunk_values = chunk_values[:, None]
        pvals = ((random_scores >= chunk_values).sum(axis=1) + 1) / (n_random + 1)
        all_pvals.append(pvals)
    all_pvals = torch.cat(all_pvals)
    return all_pvals.cpu()


def compute_pvals_torch(scores, random_scores, device):
    pvals = {}
    for target in scores:
        pvals[target] = compute_pvals_target_torch(scores[target], random_scores[target])
        torch.cuda.empty_cache()
    return pvals
