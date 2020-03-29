#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 14:40:51 2018
Modified on Thu Jun 20 11:38:06 2019

@author: Omid Sadjadi <omid.sadjadi@nist.gov>
"""

import argparse
import sre_scorer as sc
import numpy as np


# def score_me(score_list, label_list, configuration):
#     # trial_key = st.read_tsv_file(trial_key_file)
#     # sys_out = st.read_tsv_file(system_output_file)
#     scores = score_list
#     tar_nontar_labs = label_list
    
#     results = {}
#     for ds, p_target in configuration.items():
#         dataset_mask = np.array([True for _ in range(len(tar_nontar_labs))])
#         ds_scores = scores[dataset_mask]
#         ds_trial_labs = tar_nontar_labs[dataset_mask]

#         partition_masks = [dataset_mask[dataset_mask]]
#         act_c = st.compute_equalized_act_cost(ds_scores, partition_masks,
#                                               ds_trial_labs, p_target)
#         eer, min_c = st.compute_equalized_min_cost(ds_scores,
#                                                    partition_masks,
#                                                    ds_trial_labs, p_target)
#         results[ds] = [eer, min_c, act_c]
#     return results

def score_me(score_list, label_list, configuration):
    scores = score_list
    tar_nontar_labs = label_list
    
    results = {}

    p_target = configuration['p_target']
    c_miss = configuration['c_miss']
    c_fa = configuration['c_fa']
    act_c_avg = 0.
    for p_t in p_target:
        act_c, _, _ = sc.compute_actual_cost(scores,\
        tar_nontar_labs, p_t,\
        c_miss, c_fa)
        act_c_avg += act_c
    act_c_avg = act_c_avg / len(p_target)

    weights = None
    fnr, fpr = sc.compute_pmiss_pfa_rbst(scores, tar_nontar_labs, weights)
    eer = sc.compute_eer(fnr, fpr)
    avg_min_c = 0.
    for p_t in p_target:
        avg_min_c += sc.compute_c_norm(fnr, fpr, p_t)
    avg_min_c = avg_min_c / len(p_target)
    
    results['OUT'] = [eer, avg_min_c, act_c_avg]

    print('\nSet\tEER[%]\tmin_C\tact_C')
    for ds, res in results.items():
        eer, minc, actc = res
        print('{}\t{:05.2f}\t{:.3f}\t{:.3f}'.format(ds.upper(), eer*100,
              minc, actc))

    return results

def scoring(tsv_file, key):
    score_list = []
    with open(tsv_file, 'r') as f:
        for i in f:
            score_list.append(float(i.split(' ')[-1][:-1]))
    score_list = np.array(score_list, dtype=np.float)
    
    label_list = []
    with open(key, 'r') as f:
        for i in f:
            if i.split(' ')[-1][:-1] == 'target':
                label_list.append(1)
            else:
                label_list.append(0)
    label_list = np.array(label_list, dtype=np.int)

    configuration = {'p_target': [0.01, 0.005], 'c_miss': 1, 'c_fa': 1}
    # configuration = {'p_target': [0.01], 'c_miss': 1, 'c_fa': 1}

    results = score_me(score_list, label_list, configuration)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scorer.')
    parser.add_argument("-output", "--output", help="path to system output file",
                        type=str, required=True)
    parser.add_argument("-key", "--key", help="path to the list of keys, "
                        "e.g., /path/to/LDC2018E46/doc/sre18_dev_trials.tsv",
                        type=str, required=True)

    args = parser.parse_args()

    tsv_file = args.output
    key = args.key
    # tsv_file = '/Lun2/rzz/kaldi-master/egs/zhiyong/sre19/scoring_software_19/cts_challenge_scoring_software/system_output/scores'
    # tsv_file = '/Lun2/rzz/kaldi-master/egs/zhiyong/sre19/exp/Xvector_SAP_nodilate_1L_long_cosanel_testspeed/train_log/temporal_results/e7s85810end/scores_score_norm'
    # tsv_file = '/Lun2/rzz/kaldi-master/egs/zhiyong/sre19/scoring_software_19/cts_challenge_scoring_software/system_output/score_plda_360_calib'


    score_list = []
    with open(tsv_file, 'r') as f:
        for i in f:
            score_list.append(float(i.split(' ')[-1][:-1]))
    score_list = np.array(score_list, dtype=np.float)
    
    label_list = []
    with open(key, 'r') as f:
        for i in f:
            if i.split(' ')[-1][:-1] == 'target':
                label_list.append(1)
            else:
                label_list.append(0)
    label_list = np.array(label_list, dtype=np.int)

    configuration = {'p_target': [0.01, 0.005], 'c_miss': 1, 'c_fa': 1}

    _ = score_me(score_list, label_list, configuration)

