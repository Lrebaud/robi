from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
import scipy.optimize as opt
from sklearn.datasets import make_regression
import numpy as np
import pandas as pd


def check_n_random(pvals, targets, n_random):
    """
    Check if the number of random features is sufficient based on the minimum p-value.
    The number of random features that are better than the candidates should be high enough
    to ensure precise p-value estimation, event for candidates with really high score.
    Prints a warning if the number of random features is considered insufficient.
    """

    # Calculate the minimum p-value across all

    min_pval = np.min([pvals[t].min().numpy()[()] for t in pvals])
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

    # # Check if the specified columns exist in the dataframe
    # if not all(col in df.columns for col in columns):
    #     raise ValueError("One or more specified columns do not exist in the dataframe.")
    #
    # # Create a new dataframe with the normalized columns
    normalized_df = df.copy()
    #
    # # Calculate the z-score for each column
    # std = df[columns].std()
    # std[(std == 0) | (np.isnan(std))] = 1
    normalized_df.loc[:, columns] = (df[columns] - df[columns].mean()) / df[columns].std()

    return normalized_df


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
                  f"Consider adding {confounder} to the strata list.")
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
            print(
                'Could not load PyTorch. Consider installing or fixing it for substantial speed gains. Using Numpy instead. ')
        return False


def format_targets(df, targets):
    """
    :param df:
    :param targets: possible formats
        't1'
        ['t1']
        ['t1', 't2]
        {
            't1': ('time', 'event'),
        }
        {
            't1': ('time1', 'event1'),
            't2': ('time2', 'event2'),
        }
        {
            't1': ('time1'),
            't2': ('time2', 'event2'),
        }
        {
            't1': 'time1',
            't2': ('time2', 'event2'),
        }
    :return:
    """
    df = df.copy()
    invalid = False

    if isinstance(targets, str):
        df['events'] = True
        targets = {
            targets: (targets, 'events')
        }

    if isinstance(targets, list):
        new_targets = {}
        for t in targets:
            df[t + '_events'] = True
            new_targets[t] = (t, t + '_events')
        targets = new_targets

    if isinstance(targets, dict):
        for t in targets:
            if isinstance(targets[t], str):
                df[t + '_events'] = True
                targets[t] = (targets[t], t + '_events')
            elif isinstance(targets[t], tuple) or isinstance(targets[t], list):
                if len(targets[t]) != 2:
                    if len(targets[t]) == 1:
                        df[t + '_events'] = True
                        targets[t] = (targets[t], t + '_events')
                    else:
                        invalid = True
            else:
                invalid = True

    if invalid:
        raise ValueError('Invalid target format. See the README for examples.')

    return df, targets


def new_synthetic_dataset(n_samples, censoring, nb_features, n_informative, effective_rank, noise):
    X, Y, coef = make_regression(n_samples=n_samples,
                                 n_features=nb_features,
                                 n_informative=n_informative,
                                 effective_rank=effective_rank,
                                 noise=noise,
                                 coef=True)

    Y += abs(np.min(Y))

    def get_observed_time(x):
        rnd_cens = np.random.RandomState(0)
        # draw censoring times
        time_censor = rnd_cens.uniform(high=x, size=n_samples)
        event = Y < time_censor
        time = np.where(event, Y, time_censor)
        return event, time

    def censoring_amount(x):
        event, _ = get_observed_time(x)
        cens = 1.0 - event.sum() / event.shape[0]
        return (cens - censoring) ** 2

    # search for upper limit to obtain the desired censoring amount
    res = opt.minimize_scalar(censoring_amount,
                              method="bounded",
                              bounds=(0, Y.max()))

    # compute observed time
    event, time = get_observed_time(res.x)

    # upper time limit such that the probability
    # of being censored is non-zero for `t > tau`
    tau = time[event].max()
    mask = time < tau
    X = X[mask]
    Y = Y[mask]
    event = event[mask]

    df = pd.DataFrame(data=X)
    df['time'] = Y
    df['event'] = event
    df.columns = df.columns.values.astype(str)

    return df, coef
