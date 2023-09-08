from robi.evaluation import univariate_evaluation, score_of_random, get_permuted_scores
from robi.preselection import primary_selection
from robi.utils import torch_installed, format_targets, normalize_columns, check_confounders, check_n_random
from robi.utils_selection import get_permissiveness_effect, plot_permissiveness_effect, format_output
import numpy as np

try:
    import torch
except:
    print('Could not load PyTorch. Consider installing or fixing it for substantial speed gains. Using Numpy instead. ')


def _make_selection(df,
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

    # check which backend to use (PyTorch or Numpy). Always choose PyTorch if available
    use_torch = torch_installed(verbose=True)
    if use_torch and device != 'cuda':
        print(f"device is {device}. If you have a GPU, switching to cuda will provide significant speed gains.")

    # check and format arguments
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

    # normalize feature values
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

    # compute scores of random feature to assess the p-values of the real candidates
    scores_random = score_of_random(df, targets, device, n_uni_pval, use_torch)

    # compute C-index and associated p-values of candidates
    scores, pvals = univariate_evaluation(df, candidates, targets, device, scores_random, use_torch)

    # send a message if the number of random features used to calculate p-values (n_uni_pval) is too low
    if verbose:
        check_n_random(pvals, n_uni_pval, use_torch)

    # get p-values of permuted candidates (target permutation) to evaluate number of false positives
    pvals_permut = get_permuted_scores(df, candidates, scores_random, targets, device, n_fp_estimate, verbose, use_torch)

    # run the selection on both real and permuted candidates
    sel_by_permissiveness, nfp_by_permissiveness = get_permissiveness_effect(pvals, pvals_permut,
                                                                             corr_clusters, targets,
                                                                             n_jobs)
    if verbose:
        plot_permissiveness_effect(sel_by_permissiveness, nfp_by_permissiveness, targets)

    # format the output to make it usable
    sel_by_permissiveness, scores = format_output(nfp_by_permissiveness,
                                                  sel_by_permissiveness,
                                                  scores,
                                                  pvals,
                                                  candidates,
                                                  use_torch)
    return sel_by_permissiveness, scores, use_torch


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
    sel_by_permissiveness, scores, use_torch = _make_selection(**locals())

    # free GPU memory if needed
    if use_torch:
        torch.cuda.empty_cache()

    return sel_by_permissiveness, scores
