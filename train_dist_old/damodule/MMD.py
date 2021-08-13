import os
import sys
# sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
import torch
import torch.nn as nn
import numpy as np
# import models
from torch.autograd import Function
import torch.nn.functional as F
import importlib

__all__ = ["MMD"]

class MMD(nn.Module):
    def __init__(self, spk_clf_head, spk_backbone, nOut, **kwargs):
        super(MMD, self).__init__()

        self.emb_size = nOut
        self.domain_classes = 2

        ## MMD params
        self.gaussian_sig = [1, 2, 5, 10, 20, 40, 80]
        
        self.backbone = spk_backbone
        self.metrics = spk_clf_head

    
    def get_optimizer(self, optimizer, **kwargs):

        Optimizer = importlib.import_module('optimizer.'+optimizer).__getattribute__('Optimizer')
        self.opt_e_c = Optimizer(list(self.backbone.parameters())+list(self.metrics.parameters()), **kwargs)

        return self.opt_e_c

    def get_scheduler(self, scheduler, **kwargs):
        Scheduler = importlib.import_module('scheduler.'+scheduler).__getattribute__('Scheduler')
        self.sche_e_c, self.lr_step, self.expected_step = Scheduler(self.opt_e_c, **kwargs)

        return self.sche_e_c, self.lr_step, self.expected_step


    
    def forward(self, emb, y, y_d):

        assert (emb.size(0) % 2) == 0

        main_emb = torch.zeros([emb.size(0)//2, emb.size(1)], dtype=torch.float).cuda()
        target_emb = torch.zeros([emb.size(0)//2, emb.size(1)], dtype=torch.float).cuda()

        for i, per_emb in enumerate(emb):
            if int(y_d[i]) == 0:
                main_emb[i] = per_emb
            else:
                target_emb[i%(emb.size(0)//2)] = per_emb

        loss_c, acc = self.metrics(emb, y)

        # x_x_norm_2 = torch.zeros([main_emb.size(0), main_emb.size(0)], dtype=torch.float).cuda()
        # x_y_norm_2 = torch.zeros([main_emb.size(0), target_emb.size(0)], dtype=torch.float).cuda()
        # y_y_norm_2 = torch.zeros([target_emb.size(0), target_emb.size(0)], dtype=torch.float).cuda()

        MMt = torch.mm(main_emb, main_emb.t())
        MTt = torch.mm(main_emb, target_emb.t())
        TTt = torch.mm(target_emb, target_emb.t())

        x_x_norm_2 = torch.sum(torch.eye(MMt.size(0)).cuda() * (MMt), axis=1).unsqueeze(1) \
            - 2.0*MMt \
            + torch.sum(torch.eye(MMt.size(1)).cuda() * (MMt), axis=1).unsqueeze(0)

        x_y_norm_2 = torch.sum(torch.eye(MTt.size(0)).cuda() * (MMt), axis=1).unsqueeze(1) \
            - 2.0*MTt \
            + torch.sum(torch.eye(MTt.size(1)).cuda() * (TTt), axis=1).unsqueeze(0)

        y_y_norm_2 = torch.sum(torch.eye(TTt.size(0)).cuda() * (TTt), axis=1).unsqueeze(1) \
            - 2.0*TTt \
            + torch.sum(torch.eye(TTt.size(1)).cuda() * (TTt), axis=1).unsqueeze(0)


        # for i in range(main_emb.size(0)):
        #     for j in range(main_emb.size(0)):
        #         x_x_norm_2[i, j] = torch.norm(main_emb[i] - main_emb[j]) ** 2

        # for i in range(main_emb.size(0)):
        #     for j in range(target_emb.size(0)):
        #         x_y_norm_2[i, j] = torch.norm(main_emb[i] - target_emb[j]) ** 2

        # for i in range(target_emb.size(0)):
        #     for j in range(target_emb.size(0)):
        #         y_y_norm_2[i, j] = torch.norm(target_emb[i] - target_emb[j]) ** 2

        L_MMD = 0.0

        for sig in self.gaussian_sig:
            L_MMD_single = torch.mean(torch.exp((-1.0/(2 * sig)) * x_x_norm_2)) \
                        -2 * torch.mean(torch.exp((-1.0/(2 * sig)) * x_y_norm_2)) \
                        +torch.mean(torch.exp((-1.0/(2 * sig)) * y_y_norm_2))
            L_MMD += L_MMD_single

        L_MMD_return = torch.mean(L_MMD.detach())

        return [loss_c, L_MMD], [acc, L_MMD_return]
