import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import matplotlib.pyplot as plt
# from models.metrics import *

class AMSoftmax_normfree(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(AMSoftmax_normfree, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label, s, m):
        self.s = s
        self.m = m
        norm = torch.norm(input, dim=1, keepdim=True)
        nm_W = F.normalize(self.weight)
        cosine = F.linear(F.normalize(input), nm_W)
        margin = torch.zeros_like(cosine)

        for i in range(cosine.size(0)):
            lb = int(label[i])
            margin[i, lb] = self.m

        return norm * (cosine - margin), nm_W

class AM_normfree_NT(nn.Module):
    def __init__(self, backbone, model_settings):
        super(AM_normfree_NT, self).__init__()
        self.th_step = model_settings['anneal_step']       
        self.max_m = model_settings['m']
        self.model_settings = model_settings
        self.HistK_len = model_settings['HistK_len']

        self.iter = 0.0
        self.cos_bank = np.zeros(0, dtype=np.float32)
        self.u_r = None
        self.delta_r = None
        self.sig = None
        
        self.backbone = nn.DataParallel(backbone)
        self.metrics = AMSoftmax_normfree(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], s=model_settings['s'], m=model_settings['m'])
        self.loss = torch.nn.CrossEntropyLoss()
    
    def _append_queue(self, ent):
        if self.cos_bank.shape[0] < self.HistK_len:
            self.cos_bank = np.append(self.cos_bank, ent)
        else:
            self.cos_bank[:-1] = self.cos_bank[1:]
            self.cos_bank[-1] = ent
    
    def forward(self, x, y, mod):
        if mod == 'train':
            self.iter += 1.0
            m = min(self.max_m, (self.iter / self.th_step) * self.model_settings['m'])
        else:
            m = 0.0

        emb = self.backbone(x, y)
        logits, nm_W = self.metrics(emb, y, s=self.model_settings['s'], m=m)

        if mod == 'train':
            nm_emb = F.normalize(emb.data)
            cos_all = torch.mm(nm_W.data, nm_emb.t())
            cos_all_np = cos_all.data.cpu().numpy()
            label = y.data.cpu().numpy()

            cos_bank_batch = np.zeros(0, dtype=np.float32)
            for n, i in enumerate(label):
                self._append_queue(cos_all_np[i][n])
                cos_bank_batch = np.append(cos_bank_batch, cos_all_np[i][n])

            if (self.iter > self.model_settings['add_weight_steps']) and (self.iter % self.model_settings['weight_interval'] == 0) or (self.iter == self.model_settings['add_weight_steps']+1): 
                bins = np.linspace(-0.5, 1, 75)
                hist = plt.hist(self.cos_bank, bins, density=False)
                u_r = hist[1][np.argmax(hist[0])]
                delta_r = hist[1][np.where((hist[0]>self.model_settings['right_th']) == True)[0][-1]]
                # sig = (delta_r - u_r)/self.model_settings['sig_factor']
                sig = (delta_r - u_r)/self.model_settings['sig_factor']

                self.u_r = u_r.astype(np.float32)
                self.delta_r = delta_r.astype(np.float32)
                self.sig = sig.astype(np.float32)

            if self.iter > self.model_settings['add_weight_steps']:
                w_3_batch = np.e**((-(cos_bank_batch-self.u_r - 0.1)**2)/(2*self.sig**2))
                logits = logits * torch.tensor(w_3_batch[:, None]).cuda()


        loss = self.loss(logits, y)

        pred = logits.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        label = y.data.cpu().numpy()        
        acc = np.mean((pred == label).astype(int))

        return loss, logits, emb, acc, 0