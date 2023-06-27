
import pandas as pd
import numpy as np

from evidently.calculations.stattests import jensenshannon_stat_test
from evidently.calculations.stattests import kl_div_stat_test
from evidently.calculations.stattests import ks_stat_test
from evidently.calculations.stattests import psi_stat_test
from evidently.calculations.stattests import wasserstein_stat_test

def return_drift(ref: pd.Series, curr: pd.Series):
    """
    Summary: Returns the statistical difference between population of ref and curr

    Input: 
    ref:- Reference serie based on historical data
    curr:- Test series to be compared with

    Output: Series with the stats scores of different tests
    """

    report = pd.DataFrame(columns=['stat_test', 'drift_score', 'is_drifted'])
    for i, (stattest, threshold) in enumerate(
        zip([ks_stat_test, psi_stat_test, kl_div_stat_test, jensenshannon_stat_test, wasserstein_stat_test],
            [0.05, 0.1, 0.1, 0.1, 0.1])):
        report.loc[i, 'stat_test'] = stattest.display_name
        report.loc[i, 'drift_score'], report.loc[i, 'is_drifted'] = \
        stattest.func(ref, curr, 'num', threshold)
    return report['drift_score']


def return_drift_indicator(curr_ref: pd.Series, curr_buff: pd.Series, return_dataframe=False):
        
    """
    Summary: Checks for drift between reference and test series

    Input:
    curr_ref :- The reference series
    curr_buff :- Current test series
    return_dataframe :- Boolean variable indicating whether to return the results dataframe

    Output:
    curr_result :- Indicator denoting whether there is a drift or not
    curr_result :- Results dataframe with stats test values and individual indicators
    """
    
    fun_name = [ks_stat_test, psi_stat_test, kl_div_stat_test, jensenshannon_stat_test, wasserstein_stat_test]
    test_names = [each_test.display_name for each_test in fun_name]

    ## Thersholds for each test
    PSI_threshold = 3
    JS_threshhold = 0.5
    KL_Div_threshold = 2
    overall_threshold = 0.5

    curr_t = 1

    ## Result dataframe with stats test values
    curr_res = return_drift(curr_ref, curr_buff)

    ## Curr_value
    curr_check = pd.DataFrame()
    curr_check['Test'] = test_names
    curr_check['t=' + str(curr_t)] = curr_res

    curr_check = curr_check.T
    curr_check.columns = curr_check.iloc[0]
    curr_check.drop(curr_check.index[0],inplace=True)
    curr_check.reset_index(drop=False,inplace=True)

    ## Checking if there is a drift based on individual and test values 
    curr_check['PSI_drift'] = np.where(curr_check['PSI']>PSI_threshold,1,0)
    curr_check['JS_drift'] = np.where(curr_check['Jensen-Shannon distance']>JS_threshhold,1,0)
    curr_check['KL_drift'] = np.where(curr_check['Kullback-Leibler divergence']>KL_Div_threshold,1,0)

    ## Considering weighted results of tests and getting the overall result
    curr_check['Overall_drift'] = np.where(((0.4*curr_check['PSI_drift']+
                                              0.4*curr_check['JS_drift']+
                                              0.2*curr_check['KL_drift']))>overall_threshold,1,0)

    curr_result = curr_check['Overall_drift'][0]

    if return_dataframe:
        return (curr_result,curr_check)
    else:
        return curr_result
