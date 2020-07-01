# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: JFZhou 2020-05-31)

import numpy as np
import os
import sys

# sys.path.insert(0, 'subtools/pytorch')

# import libs.support.kaldi_io as kaldi_io
# from plda_base import PLDA

class CIP(object):
    """
    Reference:
    Wang Q, Okabe K, Lee K A, et al. A Generalized Framework for Domain Adaptation of PLDA in Speaker Recognition[C]//ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2020: 6619-6623.
    """
    def __init__(self, 
                 interpolation_weight=0.5):

        self.interpolation_weight = interpolation_weight

    def interpolation(self, mean_in, within_var_in, between_var_in, coral):
        

        # mean_in,between_var_in,within_var_in = self.plda_read(plda_in_domain)

        self.mean = mean_in
        self.between_var = self.interpolation_weight*coral.between_var+(1-self.interpolation_weight)*between_var_in
        self.within_var = self.interpolation_weight*coral.within_var+(1-self.interpolation_weight)*within_var_in

    # def plda_read(self,plda):
      
    #     with kaldi_io.open_or_fd(plda,'rb') as f:
    #         for key,vec in kaldi_io.read_vec_flt_ark(f):
    #             if key == 'mean':
    #                 mean = vec.reshape(-1,1)
    #                 dim = mean.shape[0]
    #             elif key == 'within_var':
    #                 within_var = vec.reshape(dim, dim)
    #             else:
    #                 between_var = vec.reshape(dim, dim)

    #     return mean,between_var,within_var

def main():

    if len(sys.argv)!=5:
        print('<plda-out-domain> <adapt-ivector-rspecifier> <plda-in-domain> <plda-adapt> \n',
            )  
        sys.exit() 

    plda_out_domain = sys.argv[1]
    train_vecs_adapt = sys.argv[2]
    plda_in_domain = sys.argv[3]
    plda_adapt = sys.argv[4]


    coral=CORAL()
    coral.plda_read(plda_out_domain)

    for _,vec in kaldi_io.read_vec_flt_auto(train_vecs_adapt):
        coral.add_stats(1,vec)
    coral.update_plda()


    cip=CIP()
    cip.interpolation(coral,plda_in_domain)

    plda_new = PLDA()
    plda_new.mean = cip.mean
    plda_new.within_var = cip.within_var
    plda_new.between_var = cip.between_var
    plda_new.get_output()
    plda_new.plda_trans_write(plda_adapt)

if __name__ == "__main__":
    main()