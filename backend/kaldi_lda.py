import scipy
import numpy as np
import math
from numpy.linalg import inv

class LDA(object):
    def __init__(self, lda_dim, ivector_dim, total_covariance_factor=0.0, covariance_floor=1.0e-06):
        self.lda_dim = lda_dim
        self.ivector_dim = ivector_dim
        self.total_covariance_factor = total_covariance_factor
        self.covariance_floor = covariance_floor

        self.tot_covar_ = np.zeros([ivector_dim, ivector_dim])
        self.between_covar_ = np.zeros([ivector_dim, ivector_dim])
        self.global_mean = np.zeros(ivector_dim)
        self.num_spk_ = 0
        self.num_utt_ = 0
    
    def AccStats(self, utts_of_this_spk):
        num_utts = utts_of_this_spk.shape[0]
        self.tot_covar_ += utts_of_this_spk.T.dot(utts_of_this_spk)
        spk_average = np.mean(utts_of_this_spk, axis=0)
        self.global_mean += num_utts * spk_average
        self.between_covar_ += num_utts * spk_average[:, None].dot(spk_average[None, :])
        self.num_utt_ += num_utts
        self.num_spk_ += 1

    def GetTotalCovar(self):
        return (1.0 / self.num_utt_) * self.tot_covar_

    def GetWithinCovar(self):      
        return (1.0 / self.num_utt_) * (self.tot_covar_ - self.between_covar_)
    
    def GetGlobalMean(self):
        mean_vector = (1.0 / self.num_utt_) * self.global_mean
        l2norm = np.linalg.norm(mean_vector)
        return mean_vector, l2norm
    
    def sort_svd(self, s, U):
        s = s[::-1]
        U = U[:, ::-1]
        return s, U
    
    def ComputeNormalizingTransform(self, mat_to_normalize):
        s, U = np.linalg.eigh(mat_to_normalize)
        s, U = self.sort_svd(s, U)
        print(s)
        floor = self.covariance_floor * s[0]
        s = s.clip(min=floor)
        s = s ** (-0.5)
        proj = (s * np.eye(s.shape[0])).dot(U.T)
        return proj

    def ComputeLdaTransform(self):
        mean_vector, mean_norm = self.GetGlobalMean()
        print('the input data has norm of mean {}'.format(mean_norm))
        total_covar = self.GetTotalCovar()
        within_covar = self.GetWithinCovar()

        mat_to_normalize = self.total_covariance_factor * total_covar + (1.0 - self.total_covariance_factor) * within_covar
        T = self.ComputeNormalizingTransform(mat_to_normalize)
        
        between_covar = total_covar - within_covar
        between_covar_proj = T.dot(between_covar).dot(T.T)

        s, U = np.linalg.eigh(between_covar_proj)
        s, U = self.sort_svd(s, U)
        print(s)

        U_part = U[:, 0:self.lda_dim]
        lda_out = U_part.T.dot(T)

        return lda_out

