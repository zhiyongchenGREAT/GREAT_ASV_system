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

__all__ = ["DAS_BL"]

class DAS_BL(nn.Module):
    def __init__(self, spk_clf_head, spk_backbone, nOut, **kwargs):
        super(DAS_BL, self).__init__()

        self.emb_size = nOut
        self.domain_classes = 2

        ## DAS params
        self.m = 2
        
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

        # print(y_d)
        for i, per_emb in enumerate(emb):
            if int(y_d[i]) == 0:
                main_emb[i] = per_emb
            else:
                target_emb[i%(emb.size(0)//2)] = per_emb

        loss_c, acc = self.metrics(emb, y)

        DAS_dist_emb = torch.zeros([emb.size(0), emb.size(1)], dtype=torch.float).cuda()
        DAS_label = torch.zeros([emb.size(0)], dtype=torch.float).cuda()

        for i, per_emb in enumerate(main_emb):
            DAS_dist_emb[i*2,:] = per_emb - main_emb[(i+1)%(emb.size(0)//2)]
            DAS_label[i*2] = 1.0
            DAS_dist_emb[i*2+1,:] = per_emb - target_emb[i]
            DAS_label[i*2+1] = 0.0

        DAS_dist = torch.norm(DAS_dist_emb, dim=1)
        ## not add neg to DAS_loss, seems make more sense
        DAS_loss = (1.0-DAS_label) * DAS_dist**2 + DAS_label * torch.clamp((self.m-DAS_dist), min=0.0)**2
        DAS_loss = torch.mean(DAS_loss)

        DAS_dist_mean = torch.mean(DAS_dist.detach()).cpu().numpy()

        # emb_d1 = self.layer_d1(emb.detach())
        # logits_d = self.layer_d2(emb_d1)

        # emb_d1_al = self.layer_d1(emb)
        # logits_d_al = self.layer_d2(emb_d1_al)

        # loss_d = self.negloss(torch.log(self.sft(logits_d)), y_d)
        # loss_d = torch.mean(loss_d)

        # loss_al = torch.mean((-1) * torch.log(self.sft(logits_d_al)), dim=1)
        # loss_al = torch.mean(loss_al)

        # pred_d = logits_d.data.cpu().numpy()
        # pred_d = np.argmax(pred_d, axis=1)
        # label_d = y_d.data.cpu().numpy()
        # acc_d = np.mean((pred_d == label_d).astype(int))

        return [loss_c, DAS_loss], [acc, DAS_dist_mean]
