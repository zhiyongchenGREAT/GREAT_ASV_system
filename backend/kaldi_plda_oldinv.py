import scipy
import numpy as np
import math
from numpy.linalg import inv
from scipy.stats import multivariate_normal


M_LOG_2PI = 1.8378770664093454835606594728112


class ClassInfo(object):
    def __init__(self, num_example=0, mean=0):
        self.num_example = num_example
        self.mean = mean


class PldaStats(object):
    def __init__(self, dim):
        self.dim = dim
        self.num_example = 0
        self.num_classes = 0
        self.sum = np.zeros(dim)
        self.offset_scatter = np.zeros([dim, dim])
        self.classinfo = list()

    def add_samples(self, group):        
        # Each row represent an utts of the same speaker.
        n = group.shape[0]
        mean = np.mean(group, axis=0)
        
        self.offset_scatter += group.T.dot(group)
        self.offset_scatter += -n * mean[:, None].dot(mean[None, :])
        
        self.classinfo.append(ClassInfo(n, mean))

        self.num_example += n
        self.num_classes += 1

        self.sum += mean

    def is_sorted(self):
        for i in range(self.num_classes-1):
            if self.classinfo[i+1].num_example < self.classinfo[i].num_example:
                return False
        return True

    def sort(self):
        for i in range(self.num_classes - 1):
            for j in range(self.num_classes - i - 1):
                if self.classinfo[j].num_example > self.classinfo[j+1].num_example:
                    self.classinfo[j], self.classinfo[j+1] = self.classinfo[j+1], self.classinfo[j]
        return



class PLDA(object):
    """
    用于PLDA计算的类:
    transform_ivector: 对数据进行转换
    log_likelihood_ratio: 对已经注册的语句与待测试的数据进行对数似然概率计算
    smooth_within_class_covariance: 类内方差平滑
    apply_transform: 对输入进行变换
    """
    def __init__(self, mean, transform, psi):
        self.mean = mean
        self.transform = transform
        self.psi = psi
        self.offset = - self.transform.dot(self.mean)
        self.dim = self.offset.shape[-1]
        self.matrix_psi = self.psi * np.eye(self.dim)
        self.I = np.eye(self.dim)
        
        self.A = inv(self.matrix_psi + self.I) - inv((self.matrix_psi + self.I) - self.matrix_psi.dot(inv(self.matrix_psi + self.I)).dot(self.matrix_psi))
        self.G = -inv(2*self.matrix_psi + self.I).dot(self.matrix_psi).dot(inv(self.I))
      
        self.sig_I = np.zeros((2*self.dim, 2*self.dim))
        self.sig_I[0:self.dim, 0:self.dim] = self.matrix_psi + self.I
        self.sig_I[self.dim:2*self.dim, self.dim:2*self.dim] = self.matrix_psi + self.I
        self.sig_I[0:self.dim, self.dim:2*self.dim] = self.matrix_psi
        self.sig_I[self.dim:2*self.dim, 0:self.dim] = self.matrix_psi
        
        self.sig_E = np.zeros((2*self.dim, 2*self.dim))
        self.sig_E[0:self.dim, 0:self.dim] = self.matrix_psi + self.I
        self.sig_E[self.dim:2*self.dim, self.dim:2*self.dim] = self.matrix_psi + self.I

        self.logdet_I = np.linalg.slogdet(self.sig_I)[1]
        self.logdet_E = np.linalg.slogdet(self.sig_E)[1]


    def transform_ivector(self, ivector, num_example, normalize_length=True, simple_length_norm=False):
        transformed_ivec = self.transform.dot(ivector) + self.offset
        if(simple_length_norm):
            normalization_factor = math.sqrt(self.dim) / np.linalg.norm(transformed_ivec)
        else:
            normalization_factor = self.get_normalization_factor(transformed_ivec, num_example)

        if normalize_length:
            transformed_ivec = normalization_factor * transformed_ivec
        
        return transformed_ivec

    # def log_likelihood_ratio(self, transform_train_ivector, num_utts, transform_test_ivector):
    #     x_t = transform_test_ivector
    #     x_e = transform_train_ivector
    #     I = self.I
    #     psi = self.matrix_psi
    #     n = num_utts
    #     like_I = multivariate_normal.pdf(x_t, (n*psi).dot(inv(n*psi+I)).dot(x_e), I + psi.dot(inv(n*psi+I))) 
    #     like_E = multivariate_normal.pdf(x_t, np.zeros(self.dim), I + psi)
    #     return np.log(like_I) - np.log(like_E)
    
    def log_likelihood_ratio(self, transform_train_ivector, num_utts, 
        transform_test_ivector):
        self.dim = transform_train_ivector.shape[-1]
        mean = np.zeros(self.dim)
        variance = np.zeros(self.dim)
        for i in range(self.dim):
            mean[i] = num_utts * self.psi[i] / (num_utts * self.psi[i] + 1.0) * transform_train_ivector[i]
            variance[i] = 1.0 + self.psi[i] / (num_utts * self.psi[i] + 1.0)
        #
        logdet = np.sum(np.log(variance))
        sqdiff = transform_test_ivector - mean
        sqdiff = np.power(sqdiff, np.full(sqdiff.shape, 2.0))
        variance = np.reciprocal(variance)
        loglike_given_class = -0.5 * (logdet + M_LOG_2PI * self.dim + np.dot(sqdiff, variance))
        #
        sqdiff = transform_test_ivector
        sqdiff = np.power(sqdiff, np.full(sqdiff.shape, 2.0))
        variance = self.psi + 1.0
        logdet = np.sum(np.log(variance))
        variance = np.reciprocal(variance)
        loglike_without_class = -0.5 * (logdet + M_LOG_2PI * self.dim + np.dot(sqdiff, variance))
        loglike_ratio = loglike_given_class - loglike_without_class
        return loglike_ratio
    
    def simple_llr_verification(self, transform_train_ivector, transform_test_ivector, complete=False):
        x_t = transform_test_ivector
        x_e = transform_train_ivector
        if complete:
            loglike_ratio = -0.5 * (self.logdet_I - self.logdet_E - x_t.dot(self.A).dot(x_t) - x_e.dot(self.A).dot(x_e) + 2*x_e.dot(self.G).dot(x_t))
        else:
            loglike_ratio =  x_t.dot(self.A).dot(x_t) + x_e.dot(self.A).dot(x_e) - 2*x_e.dot(self.G).dot(x_t)

        return loglike_ratio  

    def get_normalization_factor(self, transform_ivector, num_example):
        transform_ivector_sq = np.power(transform_ivector, np.full(transform_ivector.shape, 2.0))
        inv_covar = self.psi + 1.0/num_example
        inv_covar = 1.0 / inv_covar
        dot_prob = np.dot(inv_covar, transform_ivector_sq)
        return np.sqrt(self.dim / dot_prob)


class PldaEstimation(object):
    """
    EM迭代的类，输入为PLDAstats, 使用estimate函数训练，get_output获得PLDA模型
    """
    def __init__(self, Pldastats):
        self.stats = Pldastats
        self.dim = Pldastats.dim
        self.between_var = np.eye(self.dim)
        self.between_var_stats = np.zeros((self.dim, self.dim))
        self.between_var_count = 0
        self.within_var = np.eye(self.dim)
        self.within_var_stats = np.zeros((self.dim, self.dim))
        self.within_var_count = 0
        self.nllr_x = 0.0
        self.nllr_y = 0.0

    def estimate(self, iteration=10):
        for i in range(iteration):
            print(i+1, iteration)
            self.estimate_one_iter()
        return self.get_output()
    
    def compute_object_function_part1(self):
        within_class_count = self.stats.num_example - self.stats.num_classes
        inv_within_var = self.within_var
        _, within_logdet = np.linalg.slogdet(inv_within_var)
        inv_within_var = np.linalg.inv(inv_within_var)       
        objf = -0.5 * (within_class_count * (within_logdet + M_LOG_2PI * self.dim)
                        + np.trace(inv_within_var.dot(self.stats.offset_scatter)))
        print('part1_residual', objf/within_class_count)
        return objf

    def compute_object_function_part2(self):
        tot_objf = 0.0
        n = -1
        for i in range(np.array(self.stats.classinfo).shape[0]):
            info = self.stats.classinfo[i]
            if n != info.num_example:
                n = info.num_example
                combined_inv_var = self.between_var + (1.0 / n) * self.within_var 
                _, combined_var_logdet = np.linalg.slogdet(combined_inv_var)
                combined_inv_var = inv(combined_inv_var)
            mean = info.mean - (self.stats.sum / self.stats.num_classes)
            tot_objf += -0.5 * (combined_var_logdet + M_LOG_2PI * self.dim 
                                                + mean.T.dot(combined_inv_var).dot(mean))
        print('part2_mean', tot_objf/self.stats.num_classes)
        return tot_objf

    def compute_object_function(self):
        ans1 = self.compute_object_function_part1()
        ans2 = self.compute_object_function_part2()
        ans = ans1 + ans2
        normalized_ans = ans / self.stats.num_example
        return normalized_ans
                
    def estimate_one_iter(self):
        self.reset_per_iter_stats()
        self.get_stats_from_intraclass()
        self.get_stats_from_class_mean()

        print('normlized_obj', self.compute_object_function())

        self.estimate_from_stats()

    def init_parameters(self):
        self.within_var = np.eye(self.dim)
        self.between_var = np.eye(self.dim)

    def reset_per_iter_stats(self):
        self.within_var_stats = np.zeros((self.dim, self.dim))
        self.within_var_count = 0
        self.between_var_stats = np.zeros((self.dim, self.dim))
        self.between_var_count = 0

    def get_stats_from_intraclass(self):
        self.within_var_stats += self.stats.offset_scatter
        self.within_var_count += self.stats.num_example - self.stats.num_classes
    
    def get_stats_from_class_mean(self):
        between_var_inv = np.linalg.inv(self.between_var)
        within_var_inv = np.linalg.inv(self.within_var)
        n = -1
        for i in range(self.stats.num_classes):
            info = self.stats.classinfo[i]
            if n != info.num_example:
                n = info.num_example          
                mix_var = between_var_inv +  n * within_var_inv
                mix_var = np.linalg.inv(mix_var)
                logdet_cov_x = np.linalg.slogdet(self.between_var)[1]
                logdet_cov_y = np.linalg.slogdet(self.within_var / n)[1]
            
            m = info.mean - (self.stats.sum / self.stats.num_classes)
            w = mix_var.dot(n * within_var_inv.dot(m))
            m_w = m - w
            stats_xxT = mix_var + w[:, None].dot(w[None, :])
            stats_yyT = mix_var + m_w[:, None].dot(m_w[None, :])
            self.between_var_stats += stats_xxT
            self.between_var_count += 1
            self.within_var_stats += n * stats_yyT
            self.within_var_count += 1

            self.nllr_x += logdet_cov_x + np.trace(stats_xxT.dot(between_var_inv))
            self.nllr_y += logdet_cov_y + np.trace(stats_yyT.dot(n * within_var_inv))
        
        print('nllr_x:', self.nllr_x/self.stats.num_classes)
        print('nllr_y:', self.nllr_y/self.stats.num_classes)
        print('normalized_nllr:', (self.nllr_x+self.nllr_y)/self.stats.num_classes)
        self.nllr_x = 0.0
        self.nllr_y = 0.0

    def estimate_from_stats(self):
        self.within_var = (1.0 / self.within_var_count) * self.within_var_stats
        self.between_var = (1.0 / self.between_var_count) * self.between_var_stats

    def get_output(self):
        W_1 = inv(np.linalg.cholesky(self.within_var))
        between_var_proj = W_1.dot(self.between_var).dot(W_1.T)

        psi, A = np.linalg.eigh(between_var_proj)
        psi, A = self.sort_svd(psi, A)

        mean = self.stats.sum / self.stats.num_classes
        transform = A.T.dot(W_1)
        psi = psi

        return [mean, transform, psi]
    
    def sort_svd(self, s, U):
        s = s[::-1]
        U = U[:, ::-1]
        return s, U


class PldaUnsupervisedAdaptor(object):
    def __init__(self, dim, mean_diff_scale=1.0, within_covar_scale=0.75, between_covar_scale=0.25):
        self.num_count = 0
        self.mean_stats = np.zeros([dim])
        self.variance_stats = np.zeros([dim, dim])
        self.dim = dim
        self.mean_diff_scale = mean_diff_scale
        self.within_covar_scale = within_covar_scale
        self.between_covar_scale = between_covar_scale
    
    def add_stats(self, ivector):
        self.num_count += 1
        self.mean_stats += ivector
        self.variance_stats += ivector[:, None].dot(ivector[None, :])

    def update_plda(self, plda_mean, plda_transform, plda_psi):
        # dim = self.mean_stats.shape[0]

        mean = (1.0 / self.num_count) * self.mean_stats
        variance = (1.0 / self.num_count) * self.variance_stats - mean[:, None].dot(mean[None, :])
        
        mean_diff = mean - plda_mean
        if self.mean_diff_scale >= 0.0:
            variance = variance + self.mean_diff_scale *  mean_diff[:, None].dot(mean_diff[None, :])
        out_mean = mean

        transform_mod = plda_transform
        for i in range(self.dim):
            transform_mod[i] *= 1.0 / math.sqrt(1.0 + plda_psi[i])
        variance_proj = transform_mod.dot(variance).dot(transform_mod.T)

        s, P = np.linalg.eigh(variance_proj)
        s, P = self.sort_svd(s, P)

        W = np.zeros([self.dim, self.dim])
        B = np.zeros([self.dim, self.dim])
        for i in range(self.dim):
            W[i][i] = 1.0 / (1.0 + plda_psi[i])
            B[i][i] = plda_psi[i] / (1.0 + plda_psi[i])
        Wproj2 = P.T.dot(W).dot(P)
        Bproj2 = P.T.dot(B).dot(P)
        Ptrans = P.T
        Wproj2mod = Wproj2
        Bproj2mod = Bproj2
        for i in range(self.dim):
            if s[i] > 1.0:
                excess_eig = s[i] - 1.0
                excess_within_covar = excess_eig * self.within_covar_scale
                excess_between_covar = excess_eig * self.between_covar_scale
                Wproj2mod[i][i] += excess_within_covar
                Bproj2mod[i][i] += excess_between_covar
        combined_trans_inv = inv(Ptrans.dot(transform_mod))
        Wmod = combined_trans_inv.dot(Wproj2mod).dot(combined_trans_inv.T)
        Bmod = combined_trans_inv.dot(Bproj2mod).dot(combined_trans_inv.T)
        C_inv = inv(np.linalg.cholesky(Wmod))
        Bmod_proj = C_inv.dot(Bmod).dot(C_inv.T)
        out_psi, Q = np.linalg.eigh(Bmod_proj)
        out_psi, Q = self.sort_svd(out_psi, Q)
        out_transform = Q.T.dot(C_inv)

        return [out_mean, out_transform, out_psi]
    
    def sort_svd(self, s, U):
        s = s[::-1]
        U = U[:, ::-1]
        return s, U
        