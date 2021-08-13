import torch
import torch.nn as nn
import numpy as np
from scipy import stats
import random
import torch.nn.functional as F

class AMSoftmax_normfree(nn.Module):
    """Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.25):
        super(AMSoftmax_normfree, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features), requires_grad=True)
        nn.init.xavier_uniform_(self.weight, gain=1)

        self.learnable_m = torch.nn.Parameter(torch.empty(out_features, 1))
        nn.init.constant_(self.learnable_m, m)


    def forward(self, input, label, s, m, **kwargs):
        self.s = s
        self.m = m
        norm = torch.norm(input, dim=1, keepdim=True)
        nm_W = F.normalize(self.weight)
        cosine = F.linear(F.normalize(input), nm_W)
        margin = torch.zeros_like(cosine)

        if 'dfi' in kwargs:
            for i in range(cosine.size(0)):
                lb = int(label[i])
                margin[i, lb] = self.m * kwargs['dfi'][i]

            return norm * (cosine - margin), cosine

        elif 'delta_var' in kwargs:
            phi = torch.zeros_like(cosine)
            for i in range(cosine.size(0)):
                lb = int(label[i])
                margin[i, lb] = self.m
                now_alpha = torch.from_numpy(np.random.normal(loc=0.0, scale=kwargs['delta_var'][lb], size=cosine.size(1))).cuda(non_blocking=True) * kwargs['delta_var_s'] *self.m
                cos_alpha = torch.cos(now_alpha)
                sin_alpha = torch.sin(now_alpha)
                sine = torch.sqrt(1.0 - torch.pow(cosine[i], 2))
                phi[i] = cosine[i] * cos_alpha - sine * sin_alpha

            return norm * (phi - margin), nm_W

        elif 'learnable_margin' in kwargs:
            for i in range(cosine.size(0)):
                lb = int(label[i])
                margin[i,lb] = self.learnable_m[lb]
            learnable_m_set = self.learnable_m[label]
            return norm * (cosine - margin), self.learnable_m

        else:
            for i in range(cosine.size(0)):
                lb = int(label[i])
                margin[i, lb] = self.m

            return norm * (cosine - margin), cosine


class Center_AM_softmax(nn.Module):
    def __init__(self, nOut, nClasses, margin, scale, SBM_k, **kwargs):
        super(Center_AM_softmax, self).__init__()

        self.max_m = margin
        self.s = scale
        self.nOut = nOut
        self.num_classes = nClasses
        self.k = SBM_k
        
        self.metrics = AMSoftmax_normfree(
            in_features=self.nOut, out_features=self.num_classes, s=self.s, m=self.max_m)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.centers = torch.rand(self.num_classes, self.nOut).cuda()
        self.centers.requires_grad = False

        self.loss_rrm = torch.nn.MSELoss()

    
    def forward(self, emb3s, label_cb, **kwargs):
        # RBM
        ic, dfi = self.get_IC(label_cb, k=self.k)

        out_f, emb, dir_emb = emb3s
        # output, _ = self.metrics(emb, label_cb, s=self.s, m=self.max_m, dfi=ic)
        output, _ = self.metrics(emb, label_cb, s=self.s, m=self.max_m)
        loss = self.criterion(output, label_cb) + self.loss_rrm(out_f, dfi.unsqueeze(1))


        pred = output.detach().cpu().numpy()
        now_result = np.argmax(pred, axis=1)
        now_acc = np.mean((now_result == label_cb.detach().cpu().numpy()).astype(int))
        
        return loss, now_acc

    def get_IC(self, labels, k=10):
        self.centers = self.metrics.weight.data
        distmat = torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, self.num_classes).t()
        dismat = torch.addmm(distmat, self.centers, self.centers.t(), beta=1, alpha=-2)
        dismat_sort = torch.sort(dismat)
        nearest_index = dismat_sort[1][:,1:1+k]
        center = self.centers[labels].unsqueeze(1).expand(len(labels), k, self.centers.size(1))
        nearest_centers = self.centers[nearest_index[labels]]
        ic = torch.log(torch.exp(torch.abs((center*nearest_centers).sum(dim=2)/(torch.norm(center,dim=2)*torch.norm(nearest_centers,dim=2))))).sum(dim=1)
        dfi = 1 / ic
        ic = ic/ic.mean()
        dfi = dfi/dfi.mean()
        return ic.data, dfi.data

