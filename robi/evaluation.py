
def get_comparable_pairs(y_true):
    y = torch.from_numpy(np.ascontiguousarray(y_true['time']))
    e = torch.from_numpy(y_true['event'])

    grid_x, grid_y = torch.meshgrid(y, y)
    grid_x = grid_x.tril()
    grid_y = grid_y.tril()
    diff_truth = grid_x - grid_y

    grid_ex, grid_ey = torch.meshgrid(e, e)
    valid_pairs =((diff_truth < 0) & (grid_ex == 1))
    res1 = torch.stack(torch.where(valid_pairs)).T

    diff_truth = grid_y - grid_x
    valid_pairs =((diff_truth < 0) & (grid_ey == 1))
    res2 = torch.stack(torch.where(valid_pairs)).T.flip(1)

    pairs = torch.cat([res1,res2]).numpy()
    return np.random.permutation(pairs)

def cindex_by_pair(v_high, v_low):
    eval_comparable = (v_high < v_low).float()
    eval_non_comparable = (v_high == v_low).float()
    return (eval_comparable+(eval_non_comparable*0.5))


def compute_univariate_score_of_torch(df, target_columns, biom_values):
    univ_scores = pd.DataFrame()
    for target in target_columns:
        y = Surv.from_arrays(event=df[target_columns[target][1]].astype('bool'),
                              time=df[target_columns[target][0]])

        # get comparable train pairs values
        pairs = get_comparable_pairs(y)
        biom_values_low  = biom_values[pairs[:, 0]]
        biom_values_high = biom_values[pairs[:, 1]]

        # compute feature's signs
        features_cindex_by_pair = cindex_by_pair(biom_values_high, biom_values_low)
        features_cindex = features_cindex_by_pair.mean(dim=0)
        univ_scores[target] = features_cindex.cpu().numpy()
    return univ_scores


def compute_univariate_score(df, candidates, target_columns, device):
    max_chunk_size = 1e4
    if len(candidates) > max_chunk_size:
        n_chunks = len(candidates) / max_chunk_size
        chunks = np.array_split(candidates, n_chunks)
    else:
        chunks = [candidates]

    all_univ_scores = []
    for chunk in chunks: #tqdm(chunks, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        biom_values = torch.as_tensor(df[chunk].values, device=device, dtype=torch.float)
        univ_scores = compute_univariate_score_of_torch(df, target_columns, biom_values)
        torch.cuda.empty_cache()
        univ_scores.index = chunk
        all_univ_scores.append(univ_scores)
    return pd.concat(all_univ_scores)


def compute_pvals_target(scores, random_scores, device):
    scores = abs(scores-0.5)
    random_scores = torch.as_tensor(random_scores, device=device, dtype=torch.float)
    random_scores = abs(random_scores-0.5)
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
        pvals = ((random_scores >= chunk_values).sum(axis=1) + 1) / (n_random+1)
        all_pvals.append(pvals)
    all_pvals = torch.cat(all_pvals)
    return all_pvals.cpu().numpy()


def compute_pvals(scores, random_scores, device):
    for target in scores.columns:
        scores[target+'_pval'] = compute_pvals_target(scores[target].values, random_scores[target].values, device)
        torch.cuda.empty_cache()
    return scores

def score_of_random(df, targets, device, n_random):
    df_random = pd.DataFrame(data=np.random.uniform(0,1, (df.shape[0], int(n_random))), index=df.index)
    random_cols = df_random.columns
    df_random = pd.concat([df[sum([list(targets[x]) for x in targets], [])], df_random], axis=1)
    return compute_univariate_score(df_random, random_cols, targets, device)


def univariate_evaluation(df, candidates, targets, device, scores_random):
    scores = compute_univariate_score(df, candidates, targets, device)
    return compute_pvals(scores, scores_random, device)