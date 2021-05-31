import torch
import sys
import os
import torch
import numpy
import torch.nn.functional as F
from scipy.io import wavfile
from collections import defaultdict

def as_norm_1(file_score_path, out_path, cohort2enroll, cohort2test, top_num=200, hold_name=True, dict_sep=' '):
    # operation: norm_score = 0.5* ((score-mean_e_e)/str_e_e+(score-mean_t_t)/str_t_t)
    print('Score norm operation start, method: Adaptive_score_norm_type_1')
       
    file_scores_as_norm_path = out_path
    enroll_dict = defaultdict(list)
    test_dict = defaultdict(list)
    
    for count, i in enumerate(cohort2enroll):
        cohort_utt = i.split(dict_sep)[0]
        enroll_utt = i.split(dict_sep)[1]
        score = float(cohort2enroll[i])
        enroll_dict[enroll_utt].append([cohort_utt, score])

        if ((count+1) % 100000) == 0:
            print('read c2e:', (count+1)//100000, end='\r')

    for count, i in enumerate(cohort2test):
        cohort_utt = i.split(dict_sep)[0]
        test_utt = i.split(dict_sep)[1]
        score = float(cohort2test[i])
        test_dict[test_utt].append([cohort_utt, score])

        if ((count+1) % 100000) == 0:
            print('read c2t:', (count+1)//100000, end='\r')
        
    mean_e_e = {}
    str_e_e = {}
    mean_t_t = {}
    str_t_t = {}
    
    print('calculate all statistics')
    
    for count, key in enumerate(enroll_dict):
        enroll_dict[key] = sorted(enroll_dict[key],key = lambda x:x[1], reverse = True)
        tmp_score_list = []
        for i in range(top_num):
            tmp_score_list.append(enroll_dict[key][i][1])
        mean_e_e[key] = numpy.mean(tmp_score_list)
        str_e_e[key] = numpy.std(tmp_score_list, ddof=1)
        if ((count+1) % 1000) == 0:
            print('cal e:', (count+1)//1000, end='\r')
        
    for count, key in enumerate(test_dict):
        test_dict[key] = sorted(test_dict[key],key = lambda x:x[1], reverse = True)
        tmp_score_list = []
        for i in range(top_num):
            tmp_score_list.append(test_dict[key][i][1])
        mean_t_t[key] = numpy.mean(tmp_score_list)
        str_t_t[key] = numpy.std(tmp_score_list, ddof=1)
        if ((count+1) % 1000) == 0:
            print('cal t:', (count+1)//1000, end='\r')

    print('Scoring...')
    
    file_scores = open(file_score_path)
    with open(file_scores_as_norm_path,'w') as f:
        for count, line in enumerate(file_scores):
            enroll_utt = line.split(' ')[1].strip()
            test_utt = line.split(' ')[2].strip()
            score = float(line.split(' ')[0].strip())
            norm_score = 0.5* ((score-mean_e_e[enroll_utt])/str_e_e[enroll_utt]+(score-mean_t_t[test_utt])/str_t_t[test_utt])
            if hold_name:
                f.write('%.4f %s %s\n'%(norm_score, enroll_utt, test_utt))
            else:
                f.write('%.4f\n'%(norm_score))
            
            if ((count+1) % 10000) == 0:
                print((count+1)//10000, end='\r')
                
    file_scores.close()
    
    print('Adaptive_score_norm_type_1 is finished')


def as_norm_2(file_score_path, out_path, cohort2enroll, cohort2test, top_num=200, hold_name=True, dict_sep=' '):
    # operation: norm_score = 0.5* ((score-mean_e_t)/str_e_t+(score-mean_t_e)/str_t_e)
    print('Score norm operation start, method: Adaptive_score_norm_type_2')

    file_scores_as_norm_path = out_path
    enroll_dict = defaultdict(dict)
    test_dict = defaultdict(dict)
    
    for count, i in enumerate(cohort2enroll):
        cohort_utt = i.split(dict_sep)[0]
        enroll_utt = i.split(dict_sep)[1]
        score = float(cohort2enroll[i])
        enroll_dict[enroll_utt][cohort_utt] = score 

        if ((count+1) % 100000) == 0:
            print('read c2e:', (count+1)//100000, end='\r')      

    for count, i in enumerate(cohort2test):
        cohort_utt = i.split(dict_sep)[0]
        test_utt = i.split(dict_sep)[1]
        score = float(cohort2test[i])
        test_dict[test_utt][cohort_utt] = score

        if ((count+1) % 100000) == 0:
            print('read c2t:', (count+1)//100000, end='\r')

    for key in enroll_dict:
        tmp_tuple = sorted(enroll_dict[key].items(),key=lambda x:x[1], reverse = True)
        tmplist1 = []
        tmplist2 = []
        for i,j in tmp_tuple:
            tmplist1.append(i)
            tmplist2.append(j)
        enroll_dict[key] = dict(zip(tmplist1,tmplist2))

    for key in test_dict:
        tmp_tuple = sorted(test_dict[key].items(),key=lambda x:x[1], reverse = True)
        tmplist1 = []
        tmplist2 = []
        for i,j in tmp_tuple:
            tmplist1.append(i)
            tmplist2.append(j)
        test_dict[key] = dict(zip(tmplist1,tmplist2))
    
    print('Scoring...')

    file_scores = open(file_score_path)
    with open(file_scores_as_norm_path,'w') as f:
        for count, line in enumerate(file_scores):

            enroll_utt = line.split(' ')[1].strip()
            test_utt = line.split(' ')[2].strip()
            score = float(line.split(' ')[0].strip())
            
            cor_cohort_utt = []
            for num,(cor_cohort_utt_tmp) in enumerate(enroll_dict[enroll_utt]):
                if num<top_num:
                    cor_cohort_utt.append(cor_cohort_utt_tmp)
                else:
                    break
            tmp_score_list = []
            for tmp_utt in cor_cohort_utt:
                tmp_score_list.append(test_dict[test_utt][tmp_utt])
            mean_e_t = numpy.mean(tmp_score_list)
            str_e_t = numpy.std(tmp_score_list, ddof=1)
            
            cor_cohort_utt = []
            for num,(cor_cohort_utt_tmp) in enumerate(test_dict[test_utt]):
                if num<top_num:
                    cor_cohort_utt.append(cor_cohort_utt_tmp)
                else:
                    break
            tmp_score_list = []
            for tmp_utt in cor_cohort_utt:
                tmp_score_list.append(enroll_dict[enroll_utt][tmp_utt])
            mean_t_e = numpy.mean(tmp_score_list)
            str_t_e = numpy.std(tmp_score_list, ddof=1)
            
            norm_score = 0.5* ((score-mean_e_t)/str_e_t+(score-mean_t_e)/str_t_e)
            if hold_name:
                f.write('%.4f %s %s\n'%(norm_score, enroll_utt, test_utt))
            else:
                f.write('%.4f\n'%(norm_score))

            if ((count+1) % 10000) == 0:
                print((count+1)//10000, end='\r')
                
    print('Adaptive_score_norm_type_2 is finished')
