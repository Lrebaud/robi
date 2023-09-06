import numpy as np
import pandas as pd

try:
    import torch
except:
    print('Could not load PyTorch. Consider installing or fixing it for substantial speed gains. Using Numpy instead. ')


def get_comparable_pairs_torch(time, event):
    y = torch.from_numpy(time)
    e = torch.from_numpy(event)

    grid_x, grid_y = torch.meshgrid(y, y)
    grid_x = grid_x.tril()
    grid_y = grid_y.tril()
    diff_truth = grid_x - grid_y

    grid_ex, grid_ey = torch.meshgrid(e, e)
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
    return (eval_comparable + (eval_non_comparable * 0.5))


def compute_univariate_score_of_torch(df, target_columns, chunk, biom_values):
    univ_scores = []
    for target in target_columns:

        # get comparable train pairs values
        pairs = get_comparable_pairs_torch(df[target_columns[target][0]].values,
                                           df[target_columns[target][1]].astype('bool').values)
        biom_values_low = biom_values[pairs[:, 0]]
        biom_values_high = biom_values[pairs[:, 1]]

        # compute feature's signs
        features_cindex_by_pair = cindex_by_pair_torch(biom_values_high, biom_values_low)
        features_cindex = features_cindex_by_pair.mean(dim=0)

        univ_scores_target = pd.DataFrame(index=chunk)
        univ_scores_target['C_index'] = features_cindex.cpu().numpy()
        univ_scores_target['target'] = target
        univ_scores.append(univ_scores_target)
    return pd.concat(univ_scores)


def compute_univariate_score_torch(df, candidates, target_columns, device):
    max_chunk_size = 1e4
    if len(candidates) > max_chunk_size:
        n_chunks = len(candidates) / max_chunk_size
        chunks = np.array_split(candidates, n_chunks)
    else:
        chunks = [candidates]

    all_univ_scores = []
    for chunk in chunks:  # tqdm(chunks, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        biom_values = torch.as_tensor(df[chunk].values, device=device, dtype=torch.float)
        univ_scores = compute_univariate_score_of_torch(df, target_columns, chunk, biom_values)
        torch.cuda.empty_cache()
        all_univ_scores.append(univ_scores)
    return pd.concat(all_univ_scores)


def compute_pvals_target_torch(scores, random_scores, device):
    scores = abs(scores - 0.5)
    random_scores = torch.as_tensor(random_scores, device=device, dtype=torch.float)
    random_scores = abs(random_scores - 0.5)
    n_random = len(random_scores)

    max_chunk_size = 1e2
    if len(scores) > max_chunk_size:
        n_chunks = len(scores) / max_chunk_size
        chunks = np.array_split(scores, n_chunks)
    else:
        chunks = [scores]

    all_pvals = []
    for chunk in chunks:
        chunk_values = torch.as_tensor(chunk, device=device, dtype=torch.float)
        chunk_values = chunk_values[:, None]
        pvals = ((random_scores >= chunk_values).sum(axis=1) + 1) / (n_random + 1)
        all_pvals.append(pvals)
    all_pvals = torch.cat(all_pvals)
    return all_pvals.cpu().numpy()


def compute_pvals_torch(scores, random_scores, device):
    for target in scores['target'].unique():
        mask_target = scores['target'] == target
        random_scores_target = random_scores[random_scores['target'] == target]

        scores.loc[mask_target, 'p_value'] = compute_pvals_target_torch(scores.loc[mask_target, 'C_index'].values,
                                                                        random_scores_target['C_index'].values,
                                                                        device)
        torch.cuda.empty_cache()
    return scores
