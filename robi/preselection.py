from robi.utils_preselection import *


def primary_selection(df, candidates, confounders, strata, confound_check_corr, targets, max_corr, verbose,
                      n_workers=1):
    """Primary feature selection based on multiple criteria."""
    if verbose:
        print('\nStarting pre-selection')

    # Drop candidates with missing values
    candidates = df[candidates].dropna(axis=1).columns.tolist()
    if verbose:
        print(f"{len(candidates)} candidates remain after dropping candidates with NaNs.")

    # Drop candidates with constant values
    candidates = df[candidates].nunique().index[df[candidates].nunique() > 1].tolist()
    if verbose:
        print(f"{len(candidates)} candidates remain after dropping constant candidates.")

    # Drop candidates with low variance
    candidates = df[candidates].var()[df[candidates].var() > 1e-3].index.tolist()
    if verbose:
        print(f"{len(candidates)} candidates remain after dropping candidates with low variance.")

    if len(confound_check_corr) > 0:
        # Compute correlation between features and known confounders
        correlation_with_confounders = df[candidates].corrwith(df[confound_check_corr].max(axis=1),
                                                               method='spearman').abs()
        candidates = correlation_with_confounders[correlation_with_confounders < 0.5].index.tolist()
        if verbose:
            print(f"{len(candidates)} candidates remain after dropping candidates correlated with known.")

    if len(confound_check_corr) > 0:
        # Check multicollinearity with confounders
        all_vif = get_all_vif(df, candidates, confound_check_corr, n_workers)
        candidates = all_vif[all_vif['vif'] < 5].index.tolist()
        if verbose:
            print(
                f"{len(candidates)} candidates remain after dropping candidates with high multicollinearity with known.")

    if len(confounders) > 0:
        # Check for confoundings factor
        multivariate_eval = get_all_cox_coef(df, candidates, targets, confounders, strata, n_workers)

        candidates = multivariate_eval[
            multivariate_eval[[x for x in multivariate_eval.columns if '_diffPerct_HRnorm' in x]].abs().max(
                axis=1) < 10].index.tolist()
        if verbose:
            print(f'{len(candidates)} candidates remain after dropping candidates sensitive to confounders.')

    # Drop duplicated candidates
    candidates_correlations = df[candidates].corr(method='spearman').abs()
    candidates = drop_high_corr(candidates_correlations, max_corr=0.99)
    if verbose:
        print(f'{len(candidates)} candidates remain after dropping duplicated candidates.')

    # Group candidates into clusters of similar information based on their correlations
    corr_clusters = get_clusters(candidates, candidates_correlations, max_corr)
    if verbose:
        print(f'{len(corr_clusters)} clusters created.')

    return candidates, corr_clusters, candidates_correlations
