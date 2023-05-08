import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt


def MAP_cal(dbp, sbp):
    '''
        MAP is determined based on the SBP and DBP
        values predicted by the proposed network.
    '''
    return (sbp+2*dbp)/3


def AAMI_test(MAE_dict):
    '''
        The AAMI standard states that a device must have a
        mean difference or mean absolute error (MAE) ≤ 5 mmHg and a
        standard deviation (SD) of differences ≤ 8 mmHg for both SBP and
        DBP, with a sample size of ≥ 85 participants. Devices are assigned
        either a grade of ‘passed’ or ‘failed’, depending on whether the
        aforementioned criteria are met.
    '''
    dbp_MAE = []
    sbp_MAE = []
    data = {'dbp': [], 'sbp': []}
    max_err_dbp, max_err_sbp = 0, 0
    for key, val in MAE_dict.items():
        if val['DBP']['Test'] > max_err_dbp:
            max_err_dbp = val['DBP']['Test']
            max_pat_dbp = key
        if val['SBP']['Test'] > max_err_sbp:
            max_err_sbp = val['SBP']['Test']
            max_pat_sbp = key
        dbp_MAE.append(val['DBP']['Test'])
        sbp_MAE.append(val['SBP']['Test'])
    data['dbp'] = [round(np.mean(dbp_MAE), 2), round(np.std(dbp_MAE), 2)]
    data['sbp'] = [round(np.mean(sbp_MAE), 2), round(np.std(sbp_MAE), 2)]

    return data, (max_err_dbp, max_pat_dbp), (max_err_sbp, max_pat_sbp)

    # index = ['mean_MAE','std_MAE']
    # df = pd.DataFrame(data,index=index)
    # df.to_csv('./AAMI_result.csv')


def over_th(data, threash):
    num_over_dbp = 0
    num_over_sbp = 0
    for key, val in data.items():
        if val['DBP']['Test'] > threash:
            num_over_dbp += 1
        if val['SBP']['Test'] > threash:
            num_over_sbp += 1
    return num_over_dbp, num_over_sbp


def total_err(error_dict):
    data = error_dict
    dbp_err = []
    sbp_err = []
    pat = []
    for key, val in data.items():
        pat.append(key)
        dbp_err.append(val['DBP']['Test'])
        sbp_err.append(val['SBP']['Test'])

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(15, 9)
    output = {axs[0]: [dbp_err, 'DBP'], axs[1]: [sbp_err, 'SBP']}
    for ax in axs:
        ax.hist(output[ax][0], bins=20, rwidth=0.8)
        ax.set_xlabel('Error(mmHg)')
        ax.set_ylabel('Frequencies')
        ax.set_title(f'Histogram of error {output[ax][1]}')
        ax.grid()
    plt.savefig('./Hist_all_MAE.png')
    plt.show()


if __name__ == '__main__':
    with open('./MAE_result.json', 'r') as js:
        json_dict = json.load(js)

    data, max_dbp, max_sbp = AAMI_test(json_dict)

    print(f"Error DBP: {data['dbp'][0]} \u00B1 {data['dbp'][1]}(mmHg)")
    print(f"Error SBP: {data['sbp'][0]} \u00B1 {data['sbp'][1]}(mmHg)")
    print(f'Max MAE of DBP patient: {max_dbp[1]} with: {max_dbp[0]}(mmHg)')
    print(f'Max MAE of SBP patient: {max_sbp[1]} with: {max_sbp[0]}(mmHg)')

    Threashold = 5
    total_num = len(json_dict)
    n_dbp, n_sbp = over_th(json_dict, Threashold)

    data['dbp'].append(round(n_dbp/total_num, 2) * 100)
    data['sbp'].append(round(n_sbp/total_num, 2) * 100)
    print(f'Percentage of over {Threashold} for DBP is {data["dbp"][2]}%')
    print(f'Percentage of over {Threashold} for DBP is {data["sbp"][2]}%')
    print(f'Total Number of patients:{total_num}')

    index = ['mean_MAE(mmHg)', 'std_MAE(mmHg)', 'Percentage(%)']
    df = pd.DataFrame(data, index=index)
    df.to_csv('./AAMI_result.csv')

    total_err(json_dict)
