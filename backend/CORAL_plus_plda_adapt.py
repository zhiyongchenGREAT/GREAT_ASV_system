# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: JFZhou 2020-05-31)

import numpy as np
import os
import sys

# sys.path.insert(0, 'subtools/pytorch')

# import libs.support.kaldi_io as kaldi_io
# from plda_base import PLDA


class CORALPlus(object):
    """
    通过Add_stats将新的数据添加进来，通过update_plda进行更新
    """
    def __init__(self, 
                 plda_mean, plda_within, plda_between,
                 mean_diff_scale=1.0,
                 within_covar_scale=0.8,
                 between_covar_scale=0.8):
        self.mean = plda_mean
        self.dim = self.mean.shape[0]
        self.within_var = plda_within
        self.between_var = plda_between
        self.tot_weight = 0
        self.mean_stats = 0
        self.variance_stats = 0
        self.mean_diff_scale = 1.0
        self.mean_diff_scale = mean_diff_scale
        self.within_covar_scale = within_covar_scale
        self.between_covar_scale = between_covar_scale

    def add_stats(self, weight, ivector):
        ivector = np.reshape(ivector,(-1,1))
        if type(self.mean_stats)==int:
            self.mean_stats = np.zeros(ivector.shape)
            self.variance_stats = np.zeros((ivector.shape[0],ivector.shape[0]))
        self.tot_weight += weight
        self.mean_stats += weight * ivector
        self.variance_stats += weight * np.matmul(ivector,ivector.T)
        
    def update_plda(self,):
        
        dim = self.mean_stats.shape[0]
        #TODO:Add assert
        '''
        // mean_diff of the adaptation data from the training data.  We optionally add
        // this to our total covariance matrix
        '''
        mean = (1.0 / self.tot_weight) * self.mean_stats

        '''
        D（x）= E[x^2]-[E(x)]^2
        '''
        variance = (1.0 / self.tot_weight) * self.variance_stats - np.matmul(mean,mean.T)
        '''
        // update the plda's mean data-member with our adaptation-data mean.
        '''
        mean_diff = mean - self.mean
        variance += self.mean_diff_scale * np.matmul(mean_diff,mean_diff.T)
        self.mean = mean

        o_covariance = self.within_var + self.between_var
        eigh_o, Q_o = np.linalg.eigh(o_covariance)
        self.sort_svd(eigh_o, Q_o)

        eigh_i, Q_i = np.linalg.eigh(variance)
        self.sort_svd(eigh_i, Q_i)

        EIGH_O = np.diag(eigh_o)
        EIGH_I = np.diag(eigh_i)

        C_o = np.matmul(np.matmul(Q_o,np.linalg.inv(np.sqrt(EIGH_O))),Q_o.T)
        C_i = np.matmul(np.matmul(Q_i,np.sqrt(EIGH_I)),Q_i.T)
        A = np.matmul(C_i,C_o)
        S_w = np.matmul(np.matmul(A,self.within_var),A.T)
        S_b = np.matmul(np.matmul(A,self.between_var),A.T)

        eigh_w,Q_w = np.linalg.eigh(self.within_var)
        self.sort_svd(eigh_w, Q_w)
        eigh_diag_w = np.linalg.inv(np.diag(np.sqrt(eigh_w)))
        transform_com_w = np.matmul(eigh_diag_w,Q_w.T)
        E_w,P_w = np.linalg.eigh(np.matmul(np.matmul(transform_com_w,S_w),transform_com_w.T))
        B_w =np.matmul(np.matmul(Q_w,eigh_diag_w),P_w)


        self.within_var = self.within_var + self.within_covar_scale* np.matmul(np.matmul(np.linalg.inv(B_w).T,np.maximum(0,np.diag(E_w)-np.eye(dim))),np.linalg.inv(B_w))

        eigh_b,Q_b = np.linalg.eigh(self.between_var)
        self.sort_svd(eigh_b, Q_b)
        eigh_diag_b = np.linalg.inv(np.diag(np.sqrt(eigh_b)))
        transform_com_b = np.matmul(eigh_diag_b,Q_b.T)
        E_b,P_b = np.linalg.eigh(np.matmul(np.matmul(transform_com_b,S_b),transform_com_b.T))
        B_b =np.matmul(np.matmul(Q_b,eigh_diag_b),P_b)
        self.between_var = self.between_var + self.between_covar_scale* np.matmul(np.matmul(np.linalg.inv(B_b).T,np.maximum(0,np.diag(E_b)-np.eye(dim))),np.linalg.inv(B_b))

    def sort_svd(self,s, d):
      
        for i in range(len(s)-1):
            for j in range(i+1,len(s)):
                if s[i] > s[j]:
                    s[i], s[j] = s[j], s[i]
                    d[i], d[j] = d[j], d[i]

    # def plda_read(self,plda):
      
    #     with kaldi_io.open_or_fd(plda,'rb') as f:
    #         for key,vec in kaldi_io.read_vec_flt_ark(f):
    #             if key == 'mean':
    #                 self.mean = vec.reshape(-1,1)
    #                 self.dim = self.mean.shape[0]
    #             elif key == 'within_var':
    #                 self.within_var = vec.reshape(self.dim, self.dim)
    #             else:
    #                 self.between_var = vec.reshape(self.dim, self.dim)

    # def plda_write(self,plda):
    
    #     with kaldi_io.open_or_fd(plda,'wb') as f:
    #         kaldi_io.write_vec_flt(f, self.mean, key='mean')
    #         kaldi_io.write_vec_flt(f, self.within_var.reshape(-1,1), key='within_var')
    #         kaldi_io.write_vec_flt(f, self.between_var.reshape(-1,1), key='between_var')

def main():

    if len(sys.argv)!=4:
        print('<plda> <adapt-ivector-rspecifier> <plda-adapt> \n',
            )  
        sys.exit() 

    plda = sys.argv[1]
    train_vecs_adapt = sys.argv[2]
    plda_adapt = sys.argv[3]

    coralplus=CORALPlus()
    coralplus.plda_read(plda)

    for _,vec in kaldi_io.read_vec_flt_auto(train_vecs_adapt):
        coralplus.add_stats(1,vec)
    coralplus.update_plda()

    plda_new = PLDA()
    plda_new.mean = coralplus.mean
    plda_new.within_var = coralplus.within_var
    plda_new.between_var = coralplus.between_var
    plda_new.get_output()
    plda_new.plda_trans_write(plda_adapt)

if __name__ == "__main__":
    main()