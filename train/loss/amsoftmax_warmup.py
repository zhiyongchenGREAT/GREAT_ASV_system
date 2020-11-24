#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Adapted from https://github.com/CoinCheung/pytorch-loss (MIT License)

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from utils import accuracy

class LossFunction(nn.Module):
    def __init__(self, nOut, nClasses, margin=0.3, scale=15, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True
        
        self.m = margin
        self.s = scale
        self.in_feats = nOut
        self.W = torch.nn.Parameter(torch.randn(nOut, nClasses), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

        print('Initialised AMSoftmax m=%.3f s=%.3f'%(self.m,self.s))

        self.step = 0

    def forward(self, x, label=None):

        self.step += 1
        if self.step <= 10000:
            mod_m = self.m * (self.step / 10000.0)
        else:
            mod_m = self.m

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats

        x_norm_scale = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm_scale)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        label_view = label.view(-1, 1)
        if label_view.is_cuda: label_view = label_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, mod_m)
        if x.is_cuda: delt_costh = delt_costh.cuda()
        costh_m = costh - delt_costh
        costh_m_s = x_norm_scale * costh_m
        loss = self.ce(costh_m_s, label)
        prec1, _    = accuracy(costh_m_s.detach().cpu(), label.detach().cpu(), topk=(1, 5))
        return loss, prec1

