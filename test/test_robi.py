import pandas as pd
import robi

def test_dlbcl():
    df = pd.read_csv('../data/DLBCL_test.csv')

    targets = {
      'PFS': ('PFS_months', 'PFS_event_happened'),
      'OS': ('OS_months', 'OS_event_happened'),
    }
    confounders = ['ECOG', 'aaIPI', 'treatment', 'sex']
    strata = ['treatment']

    res, _ = robi.make_selection(df,
                    candidates=['TMTV', 'Dmax'],
                    targets = targets,
                    confounders=confounders,
                    strata =strata,
                    verbose=False)
    assert res[res['n_sel_PFS']>0].iloc[0]['n_sel_PFS'] == 2


    res, _ = robi.make_selection(df,
                    candidates=['TMTV', 'Dmax'],
                    targets = targets,
                    confounders=['TMTV']+confounders,
                    strata =strata,
                    verbose=False)
    assert res[res['n_sel_PFS']>0].iloc[0]['n_sel_PFS'] == 1