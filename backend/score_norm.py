# implement of Analysis of score normalization in multilingual speaker recognition: adaptive S-norm2
import numpy as np
import os
from collections import defaultdict



def score_norm(file_scores_path, file_scores_norm_path):
    #file_scores_path = os.path.join(file_scores_root,file_scores_name)
    # file_scores_norm_path = file_scores_path+'_score_norm'
    file_scores = open(file_scores_path)
    enroll_dict = defaultdict(list)
    test_dict = defaultdict(list)

    #生成每句注册句与测试句的所有分数
    for line in file_scores:
        enroll_utt = line.split(' ')[0].strip()
        test_utt = line.split(' ')[1].strip()
        score_e_t = float(line.split(' ')[2].strip())
        enroll_dict[enroll_utt].append(score_e_t)
        test_dict[test_utt].append(score_e_t)

    max_cohort = 200
    enroll_dict_mean = dict() 
    enroll_dict_std = dict()
    for key in enroll_dict:
        key_value_num = len(enroll_dict[key])
        tmp_array = np.zeros(key_value_num)
        flag = 0
        for value in enroll_dict[key]:
            tmp_array[flag]=value
            flag += 1
        ##
        max_cohort = int(flag*0.5)
        ##
        tmp_array_sorted = np.sort(tmp_array,axis=-1,kind='quicksort',order=None)[-max_cohort:]
        mean_value = np.mean(tmp_array_sorted)
        std_value = np.std(tmp_array_sorted, ddof=1)
        enroll_dict_mean[key] = mean_value
        enroll_dict_std[key] = std_value

    test_dict_mean = dict() 
    test_dict_std = dict()
    for key in test_dict:
        key_value_num = len(test_dict[key])
        tmp_array = np.zeros(key_value_num)
        flag = 0
        for value in test_dict[key]:
            tmp_array[flag]=value
            flag += 1
        # max_cohort = int(flag*0.5)
        tmp_array_sorted = np.sort(tmp_array,axis=-1,kind='quicksort',order=None)[-max_cohort:]
        mean_value = np.mean(tmp_array_sorted)
        std_value = np.std(tmp_array_sorted, ddof=1)
        test_dict_mean[key] = mean_value
        if str(std_value) == 'nan':
            std_value = 1
        test_dict_std[key] = std_value

    file_scores = open(file_scores_path)
    with open(file_scores_norm_path,'w') as f:
        for line in file_scores:
            enroll_utt = line.split(' ')[0].strip()
            test_utt = line.split(' ')[1].strip()
            score_e_t = float(line.split(' ')[2].strip())
            score_as_norm2 = 0.5*((score_e_t-test_dict_mean[test_utt])/test_dict_std[test_utt]+(score_e_t-enroll_dict_mean[enroll_utt])/enroll_dict_std[enroll_utt])
            f.write(enroll_utt+' '+test_utt+' '+str(score_as_norm2)+'\n')

if __name__ == "__main__":
    file_scores_path = '/Lun2/rzz/kaldi-master/egs/zhiyong/sre19/sdsvc/score_cosine_370_c'
    file_scores_norm_path = '/Lun2/rzz/kaldi-master/egs/zhiyong/sre19/sdsvc/score_cosine_370_norm_c'
    score_norm(file_scores_path, file_scores_norm_path)
