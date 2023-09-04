from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test


def check_n_random(scores, targets, n_random):
    """
    Check if the number of random features is sufficient based on the minimum p-value.
    The number of random features that are better than the candidates should be high enough
    to ensure precise p-value estimation, event for candidates with really high score.
    Prints a warning if the number of random features is considered insufficient.
    """

    # Calculate the minimum p-value across all targets
    min_pval = scores[[f"{t}_pval" for t in targets]].min().min()
    n_sup = int(min_pval * n_random)

    # Check if the number of superior random features is less than 200
    if n_sup < 200:
        if n_sup > 1:
            recommended_n_random = int(200 / n_sup * n_random)
            print(f"Warning: only {n_sup} random features were better than the best candidate."
                  f"\nConsider increasing n_random. We recommend at least {recommended_n_random:.2E} random features.")
        else:
            print('Warning: no random features were better than the best candidate. Consider increasing n_random.')
    else:
        print('n_random is high enough')


def normalize_columns(df, columns):
    """
    Normalize (z-score) specified columns in the dataframe.
    """

    df.loc[:, columns] = (df[columns] - df[columns].mean()) / df[columns].std()
    return df


def check_proportional_hazard_for_target(df, target, targets, confounders, strata):
    """
    Check proportional hazard assumption for a specific target.
    """

    # Construct column names based on target
    duration_col = targets[target][0]
    event_col = targets[target][1]

    # Filter necessary columns from dataframe
    df_filtered = df[confounders + [duration_col, event_col]]

    # Fit the Cox Proportional Hazard model
    cph = CoxPHFitter()
    cph.fit(df_filtered, duration_col=duration_col, event_col=event_col, strata=strata, robust=True)

    # Test for proportional hazard assumption
    results = proportional_hazard_test(cph, df_filtered)

    # Check if any confounder violates the assumption
    for confounder, row in results.summary.iterrows():
        if row['p'] < 0.05:
            print(f"{confounder} violates proportional hazard assumptions for {target} (p<{row['p']:.2f}).",
                  "Consider adding to the strata list.")
            return False

    return True


def check_proportional_hazard(df, confounders, targets, strata):
    """
    Verify the proportional hazard assumption for specified targets and confounders.
    Prints the results of the verification.
    """

    all_tests_ok = all(
        check_proportional_hazard_for_target(df, target, targets, confounders, strata) for target in targets)

    if all_tests_ok:
        print('Confounders follow proportional hazard assumptions')


def check_confounders(df, confounders, targets, strata):
    """
    Check if the number of confounders exceeds the allowed limit for each target and
    verify if the confounders follows the proportional hazard assumption.
    """

    print('\nVerifying confounders settings...')

    # Calculate the maximum allowed number of confounders for each target
    # Use the rule-of-thumb stating that you should have at least
    for target, (_, event_col) in targets.items():
        n_event = df[event_col].sum()
        max_n_conf = int(n_event / 8 + 0.5)

        if len(confounders) > max_n_conf:
            print(f'Too many confounders for {target}. Max recommended: {max_n_conf}')

    # Check for proportional hazards (function not provided in the original code)
    check_proportional_hazard(df, confounders, targets, strata)


def torch_installed(verbose=False):
    try:
        import torch
        return True
    except:
        if verbose:
            print('Could not load PyTorch. Consider installing or fixing it for substantial speed gains. Using Numpy instead. ')
        return False