import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from matplotlib import pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
from joblib import Parallel, delayed


def get_vif(df, cov):
    """
    Compute the variance inflation factor (VIF) for a given set of columns.
    """

    sdf = df[cov].dropna(how='any', axis=0)
    vif_data = {
        "feature": sdf.columns,
        "VIF": [variance_inflation_factor(sdf.values, i) for i in range(sdf.shape[1])]
    }
    max_vif = max(vif_data["VIF"])
    return {'vif': max_vif, 'candidate': cov[-1]}


def get_all_vif(df, candidates, confounders, n_jobs):
    """
    Divide the task among workers and get VIF values.
    """
    res = Parallel(n_jobs=n_jobs)(delayed(get_vif)(df, confounders+[x]) for x in candidates)
    return pd.DataFrame(res).set_index('candidate')


def compute_cox_coef(df, candidate, targets, confounders, strata):
    """
    Compute univariate and multivariate cox coefficients (hazard ratios) for the given candidate.
    """
    result = pd.DataFrame(index=[candidate])

    for target, (duration_col, event_col) in targets.items():
        cph = CoxPHFitter()

        # Univariate analysis
        uni_vars = [candidate, duration_col, event_col]
        cph.fit(df[uni_vars], duration_col=duration_col, event_col=event_col)
        coef_uni = cph.summary.loc[candidate, 'coef']
        ph_test_uni = proportional_hazard_test(cph, df[uni_vars]).summary['p'].min()

        # Multivariate analysis
        multi_vars = uni_vars + confounders
        cph.fit(df[multi_vars], duration_col=duration_col, event_col=event_col, strata=strata, robust=True)
        coef_multi = cph.summary.loc[candidate, 'coef']
        ph_test_multi = proportional_hazard_test(cph, df[multi_vars]).summary['p'].min()

        # Store results
        result[target + '_uni'] = coef_uni
        result[target + '_multi'] = coef_multi
        result[target + '_uni_HRnorm'] = np.exp(abs(coef_uni))
        result[target + '_multi_HRnorm'] = np.exp(abs(coef_multi))
        result[target + '_ph_assumpt_uni'] = ph_test_uni
        result[target + '_ph_assumpt_multi'] = ph_test_multi
        result[target + '_diff_HRnorm'] = result[target + '_multi_HRnorm'] - result[target + '_uni_HRnorm']
        result[target + '_diffPerct_HRnorm'] = (result[target + '_diff_HRnorm'] / result[
            target + '_multi_HRnorm']) * 100

    return result

def compute_cox_coef_safe(df, candidate, targets, confounders, strata):
    try:
        return compute_cox_coef(df, candidate, targets, confounders, strata)
    except:
        return None


def get_all_cox_coef(df, candidates, targets, confounders, strata, n_jobs):
    """
    Compute univariate and multivariate cox coefficients (hazard ratios) for the given candidate in parallel.
    Compute metrics to decide if a candidate is sensitive to confounders.
    """
    results = Parallel(n_jobs=n_jobs)(delayed(compute_cox_coef_safe)(df, x, targets, confounders, strata) for x in candidates)
    results = [x for x in results if x != None]
    return pd.concat(results)


def drop_high_corr(candidates_correlations, max_corr):
    """
    Randomly drop candidates that are too correlated to other candidates
    """

    upper = np.tril(np.full(candidates_correlations.shape, np.nan), k=0) + candidates_correlations
    nb_corrwith = (upper >= max_corr).sum()
    return nb_corrwith[nb_corrwith == 0].index.tolist()


def get_clusters(selected, candidates_correlations, max_corr):
    """
    Cluster selected items based on their correlations.
    """

    # If there is 0 or 1 items, return them as a single cluster
    if len(selected) < 2:
        return [selected]

    if max_corr == 1:
        # return [[x] for x in selected]
        return [[x] for x in np.arange(len(selected))]

    # Calculate distances from correlations and convert to a format suitable for linkage
    distances = 1 - candidates_correlations.loc[selected, selected]
    dist_array = ssd.squareform(distances)

    # Perform hierarchical clustering and get the optimal leaf order
    Z = linkage(dist_array, 'ward')
    order = leaves_list(optimal_leaf_ordering(Z, dist_array))

    # Order the selected items based on the clustering
    # ordered_sel = np.array(selected)[order].tolist()
    ordered_sel = order

    # Create clusters based on the correlation threshold
    clusters = []
    current_cluster = [ordered_sel[0]]

    for fid in range(1, len(ordered_sel)):
        corr = candidates_correlations.loc[selected[ordered_sel[fid]], selected[ordered_sel[fid - 1]]]
        if corr > max_corr:
            current_cluster.append(ordered_sel[fid])
        else:
            clusters.append(current_cluster)
            current_cluster = [ordered_sel[fid]]

    clusters.append(current_cluster)
    return clusters


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

    return candidates, corr_clusters, candidates_correlations
