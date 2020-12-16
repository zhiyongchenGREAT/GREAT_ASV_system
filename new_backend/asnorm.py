import numpy as np
import os
from collections import defaultdict

class Score_Helper(object):
    def __init__(self, file_score_path):
        # self_score_norm只需要由trials进行打分的文件即可，
        # 其他三个方法需要将enroll和test的话分别对同一个第三方集(cohort)进行一对一打分，分别生成
        # cohort_enroll_path和cohort_test_path， 每句enroll或test至少要对应200句以上cohort_utt
        # 后三个方法会消耗很大的计算量
        # 可以尝试先用as-norm，再self-norm
        self.file_score_path = file_score_path

    def self_score_norm(self, max_top=200, mode = 'S-norm', ratio=None, new_score_file_path = None):
        # file_score_path: trials 的打分， 建议把ratio设置为0.5（标准版本应该为1）
        # mode可设置为'S-norm','Z-norm','T-norm'
        print('Score norm operation start, method: self_score_norm_'+mode)
        if new_score_file_path:
            old_path = self.file_score_path
            self.file_score_path = new_score_file_path
        file_scores_norm_path = self.file_score_path+'_self_score_norm'
        file_scores = open(self.file_score_path)
        enroll_dict = defaultdict(list)
        test_dict = defaultdict(list)

        for line in file_scores:
            enroll_utt = line.split(' ')[0].strip()
            test_utt = line.split(' ')[1].strip()
            score_e_t = float(line.split(' ')[2].strip())
            enroll_dict[enroll_utt].append(score_e_t)
            test_dict[test_utt].append(score_e_t)

        enroll_dict_mean = dict() 
        enroll_dict_std = dict()
        for key in enroll_dict:
            key_value_num = len(enroll_dict[key])
            tmp_array = np.zeros(key_value_num)
            if max_top>key_value_num:
                max_top = key_value_num
            flag = 0
            for value in enroll_dict[key]:
                tmp_array[flag]=value
                flag += 1
            if ratio:
                max_top = int(flag*ratio)
            tmp_array_sorted = np.sort(tmp_array,axis=-1,kind='quicksort',order=None)[-max_top:]
            mean_value = np.mean(tmp_array_sorted)
            std_value = np.std(tmp_array_sorted, ddof=1)
            enroll_dict_mean[key] = mean_value
            if str(std_value) == 'nan':
                std_value = 1
            if std_value == 0:
                std_value += 0.000001
            enroll_dict_std[key] = std_value

        test_dict_mean = dict() 
        test_dict_std = dict()
        for key in test_dict:
            key_value_num = len(test_dict[key])
            tmp_array = np.zeros(key_value_num)
            if max_top>key_value_num:
                max_top = key_value_num
            flag = 0
            for value in test_dict[key]:
                tmp_array[flag]=value
                flag += 1
            if ratio:
                max_top = int(flag*ratio)
            tmp_array_sorted = np.sort(tmp_array,axis=-1,kind='quicksort',order=None)[-max_top:]
            mean_value = np.mean(tmp_array_sorted)
            std_value = np.std(tmp_array_sorted, ddof=1)
            test_dict_mean[key] = mean_value
            if str(std_value) == 'nan':
                std_value = 1
            if std_value == 0:
                std_value += 0.000001
            test_dict_std[key] = std_value
        
        file_scores = open(self.file_score_path)
        with open(file_scores_norm_path,'w') as f:
            for line in file_scores:
                enroll_utt = line.split(' ')[0].strip()
                test_utt = line.split(' ')[1].strip()
                score_e_t = float(line.split(' ')[2].strip())
                if mode == 'S-norm':
                    norm_score = 0.5*((score_e_t-test_dict_mean[test_utt])/test_dict_std[test_utt]+(score_e_t-enroll_dict_mean[enroll_utt])/enroll_dict_std[enroll_utt])
                elif mode == 'Z-norm':
                    norm_score = (score_e_t-enroll_dict_mean[enroll_utt])/enroll_dict_std[enroll_utt]
                elif mode == 'T-norm':
                    norm_score = (score_e_t-test_dict_mean[test_utt])/test_dict_std[test_utt]
                else:
                    print('No mode ERROR')
                f.write(enroll_utt+' '+test_utt+' '+str(norm_score)+'\n')
        if new_score_file_path:
            self.file_score_path = old_path
        print('Self_score_norm is finished, using '+mode)

    def s_norm(self, cohort_enroll_path, cohort_test_path, mode='S-norm'):
        # mode可设置为'S-norm','Z-norm','T-norm'
        print('Score norm operation start, method: original_score_norm_'+mode)
        file_scores_as_norm_path = self.file_score_path + '_score_norm'
        enroll_lines = open(cohort_enroll_path)
        enroll_score = []
        for line in enroll_lines:
            score = float(line.split(' ')[2].strip())
            enroll_score.append(score)
        enroll_mean = np.mean(enroll_score)
        enroll_std = np.std(enroll_score, ddof=1)
        if str(enroll_std) == 'nan':
            enroll_std = 1
        if enroll_std == 0:
            enroll_std += 0.000001
        
        test_lines = open(cohort_test_path)
        test_score = []
        for line in test_lines:
            score = float(line.split(' ')[2].strip())
            test_score.append(score)
        test_mean = np.mean(test_score)
        test_std = np.std(test_score, ddof=1)
        if str(test_std) == 'nan':
            test_std = 1
        if test_std == 0:
            test_std += 0.000001
        
        file_scores = open(self.file_score_path)
        with open(file_scores_as_norm_path,'w') as f:
            for line in file_scores:
                enroll_utt = line.split(' ')[0].strip()
                test_utt = line.split(' ')[1].strip()
                score_e_t = float(line.split(' ')[2].strip())
                if mode == 'S-norm':
                    norm_score = 0.5*((score_e_t-enroll_mean)/enroll_std+(score_e_t-test_mean)/test_std)
                elif mode == 'Z-norm':
                    norm_score = (score_e_t-enroll_mean)/enroll_std
                elif mode == 'T-norm':
                    norm_score = (score_e_t-test_mean)/test_std
                else:
                    print('No mode ERROR')
                f.write(enroll_utt+' '+test_utt+' '+str(norm_score)+'\n')
        print('Original_score_norm is finished, using '+mode)

    def as_norm_1(self, cohort_enroll_path, cohort_test_path, top_num=200, hold_name = True):
        # operation: norm_score = 0.5* ((score-mean_e_e)/str_e_e+(score-mean_t_t)/str_t_t)
        print('Score norm operation start, method: Adaptive_score_norm_type_1')
        file_scores_as_norm_path = self.file_score_path + '_adapt1_score_norm'
        enroll_dict = defaultdict(list)
        test_dict = defaultdict(list)
        enroll_lines = open(cohort_enroll_path)
        for line in enroll_lines:
            cohort_utt = line.split(' ')[0].strip()
            enroll_utt = line.split(' ')[1].strip()
            score = float(line.split(' ')[2].strip())
            enroll_dict[enroll_utt].append([cohort_utt, score])
        test_lines = open(cohort_test_path)
        for line in test_lines:
            cohort_utt = line.split(' ')[0].strip()
            test_utt = line.split(' ')[1].strip()
            score = float(line.split(' ')[2].strip())
            test_dict[test_utt].append([cohort_utt, score])
        mean_e_e = {}
        str_e_e = {}
        mean_t_t = {}
        str_t_t = {}
        for key in enroll_dict:
            enroll_dict[key] = sorted(enroll_dict[key],key = lambda x:x[1], reverse = True)
            tmp_score_list = []
            for i in range(top_num):
                tmp_score_list.append(enroll_dict[key][i][1])
            mean_e_e[key] = np.mean(tmp_score_list)
            str_e_e[key] = np.std(tmp_score_list, ddof=1)
        for key in test_dict:
            test_dict[key] = sorted(test_dict[key],key = lambda x:x[1], reverse = True)
            tmp_score_list = []
            for i in range(top_num):
                tmp_score_list.append(test_dict[key][i][1])
            mean_t_t[key] = np.mean(tmp_score_list)
            str_t_t[key] = np.std(tmp_score_list, ddof=1)
        file_scores = open(self.file_score_path)
        with open(file_scores_as_norm_path,'w') as f:
            for line in file_scores:
                enroll_utt = line.split(' ')[0].strip()
                test_utt = line.split(' ')[1].strip()
                score = float(line.split(' ')[2].strip())
                norm_score = 0.5* ((score-mean_e_e[enroll_utt])/str_e_e[enroll_utt]+(score-mean_t_t[test_utt])/str_t_t[test_utt])
                if hold_name:
                    f.write(enroll_utt+' '+test_utt+' '+str(norm_score)+'\n')
                else:
                    f.write(str(norm_score)+'\n')
        print('Adaptive_score_norm_type_1 is finished')

    def as_norm_2(self, cohort_enroll_path, cohort_test_path,top_num=200,hold_name=True):
        # operation: norm_score = 0.5* ((score-mean_e_t)/str_e_t+(score-mean_t_e)/str_t_e)
        print('Score norm operation start, method: Adaptive_score_norm_type_2')
        file_scores_as_norm_path = self.file_score_path + '_adapt2_score_norm'
        enroll_dict = defaultdict(dict)
        test_dict = defaultdict(dict)
        
        enroll_lines = open(cohort_enroll_path)
        for line in enroll_lines:
            cohort_utt = line.split(' ')[0].strip()
            enroll_utt = line.split(' ')[1].strip()
            score = float(line.split(' ')[2].strip())
            enroll_dict[enroll_utt][cohort_utt] = score
        test_lines = open(cohort_test_path)
        for line in test_lines:
            cohort_utt = line.split(' ')[0].strip()
            test_utt = line.split(' ')[1].strip()
            score = float(line.split(' ')[2].strip())
            test_dict[test_utt][cohort_utt] = score
        
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
        
        file_scores = open(self.file_score_path)
        with open(file_scores_as_norm_path,'w') as f:
            for line in file_scores:
                enroll_utt = line.split(' ')[0].strip()
                test_utt = line.split(' ')[1].strip()
                score = float(line.split(' ')[2].strip())
                cor_cohort_utt = []
                for num,(cor_cohort_utt_tmp) in enumerate(enroll_dict[enroll_utt]):
                    if num<top_num:
                        cor_cohort_utt.append(cor_cohort_utt_tmp)
                    else:
                        break
                tmp_score_list = []
                for tmp_utt in cor_cohort_utt:
                    tmp_score_list.append(test_dict[test_utt][tmp_utt])
                mean_e_t = np.mean(tmp_score_list)
                str_e_t = np.std(tmp_score_list, ddof=1)
                cor_cohort_utt = []
                for num,(cor_cohort_utt_tmp) in enumerate(test_dict[test_utt]):
                    if num<top_num:
                        cor_cohort_utt.append(cor_cohort_utt_tmp)
                    else:
                        break
                tmp_score_list = []
                for tmp_utt in cor_cohort_utt:
                    tmp_score_list.append(enroll_dict[enroll_utt][tmp_utt])
                mean_t_e = np.mean(tmp_score_list)
                str_t_e = np.std(tmp_score_list, ddof=1)
                norm_score = 0.5* ((score-mean_e_t)/str_e_t+(score-mean_t_e)/str_t_e)
                if hold_name:
                    f.write(enroll_utt+' '+test_utt+' '+str(norm_score)+'\n')
                else:
                    f.write(str(norm_score)+'\n')
        print('Adaptive_score_norm_type_2 is finished')