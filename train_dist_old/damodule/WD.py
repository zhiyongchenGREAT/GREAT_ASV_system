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

__all__ = ["WD"]

class WD(nn.Module):
    def __init__(self, spk_clf_head, spk_backbone, nOut, domain_classes, ori_weight_dict, **kwargs):
        super(WD, self).__init__()
        # self.th_step = model_settings['anneal_steps']
        # self.iter = 0.0
        # self.max_m = model_settings['m']
        self.emb_size = nOut
        self.domain_classes = domain_classes

        ## WD params
        self.gamma = 10.0
        
        # self.backbone = nn.DataParallel(Xvector_SAP_1L(model_settings['in_feat'], model_settings['emb_size']))
        self.backbone = spk_backbone
        # self.metrics = AMSoftmax_normfree(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], s=model_settings['s'], m=model_settings['m'])
        self.metrics = spk_clf_head

        self.layer_d = torch.nn.Sequential()
        self.layer_d.add_module('linear', nn.Linear(self.emb_size, self.emb_size))
        self.layer_d.add_module('relu', nn.ReLU(True))
        self.layer_d.add_module('linear', nn.Linear(self.emb_size, self.emb_size))
        self.layer_d.add_module('relu', nn.ReLU(True))
        self.layer_d.add_module('linear', nn.Linear(self.emb_size, 1))
        # self.layer_d.add_module('sigmoid', nn.Sigmoid())

        assert len(ori_weight_dict) == domain_classes

        # ori_weight_dict2 = {}
        # for i in list(ori_weight_dict.keys()):
        #     ori_weight_dict2[int(i)] = int(ori_weight_dict[i])

        # total_weight = 0
        # for i in ori_weight_dict2:
        #     total_weight += ori_weight_dict2[i]
        # # weight_split = np.array(self.model_settings['weight_split'])
        # self.weight_dict = {}
        # for i in ori_weight_dict2:
        #     self.weight_dict[i] = total_weight / (len(ori_weight_dict2)*ori_weight_dict2[i])
    
    def get_optimizer(self, optimizer, **kwargs):
        # opt_e_c = torch.optim.SGD(list(self.backbone.parameters())+list(self.metrics.parameters()), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        # opt_c = torch.optim.SGD(self.metrics.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        # opt_d = torch.optim.SGD(list(self.layer_d1.parameters())+list(self.layer_d2.parameters()), lr=1e-2, momentum=0.9, weight_decay=5e-4)

        Optimizer = importlib.import_module('optimizer.'+optimizer).__getattribute__('Optimizer')
        self.opt_e_c = Optimizer(list(self.backbone.parameters())+list(self.metrics.parameters()), **kwargs)

        opt_d_args = kwargs.copy()
        opt_d_args.pop('weight_decay')

        self.opt_d = Optimizer(list(self.layer_d.parameters()), weight_decay=0.0, **opt_d_args)

        # print(self.opt_e_c)
        # print(self.opt_d)

        return self.opt_e_c, self.opt_d

    def get_scheduler(self, scheduler, **kwargs):
        Scheduler = importlib.import_module('scheduler.'+scheduler).__getattribute__('Scheduler')
        self.sche_e_c, self.lr_step, self.expected_step = Scheduler(self.opt_e_c, **kwargs)
        self.sche_d, _, _ = Scheduler(self.opt_d, **kwargs)

        return self.sche_e_c, self.sche_d, self.lr_step, self.expected_step
    
    def forward(self, emb, y, y_d):

        assert (emb.size(0) % 2) == 0
        assert (y_d[:(y_d.size(0)//2)] == 0).all() and (y_d[(y_d.size(0)//2):] == 1).all()

        # main_emb = torch.zeros([emb.size(0)//2, emb.size(1)], dtype=torch.float).cuda()
        # target_emb = torch.zeros([emb.size(0)//2, emb.size(1)], dtype=torch.float).cuda()

        # for i, per_emb in enumerate(emb):
        #     if int(y_d[i]) == 0:
        #         main_emb[i] = per_emb
        #     else:
        #         target_emb[i%(emb.size(0)//2)] = per_emb
        
        # print(main_emb)
        # print(target_emb)

        loss_c, acc = self.metrics(emb, y)

        emb_ford = emb.detach()
        emb_ford_s = emb_ford[:(y_d.size(0)//2)]
        emb_ford_t = emb_ford[(y_d.size(0)//2):]

        epsilons = torch.rand(y_d.size(0)//2, 1).cuda()
        emb_ford_inter = epsilons * emb_ford_s + (1.0 - epsilons) * emb_ford_t
        emb_ford_inter = emb_ford_inter.detach()
        emb_ford_inter.requires_grad = True
        # emb_ford_inter.grad = torch.zeros(torch.size(emb_ford_inter))
        emb_ford_inter_o = self.layer_d(emb_ford_inter)
        # emb_ford_inter_d1 = self.layer_d1_1(emb_ford_inter_d1)
        # emb_ford_inter_d2 = self.layer_d2(emb_ford_inter_d1)
        # sum_inter_out = torch.sum(emb_ford_inter_d2)
        # sum_inter_out.backward(retain_graph=True)
        gradients = torch.autograd.grad(outputs=emb_ford_inter_o, inputs=emb_ford_inter, 
                    grad_outputs=torch.ones(emb_ford_inter_o.size()).cuda(), create_graph=True, retain_graph=True)
        gradients_finter = gradients[0]
        L_grad = (torch.norm(gradients_finter, dim=1, keepdim=False) - 1.0) ** 2

        # emb_d1 = self.layer_d1(emb_ford)
        # emb_d1 = self.layer_d1_1(emb_d1)
        emb_d2 = self.layer_d(emb_ford)
        emb_d2_s = emb_d2[:(y_d.size(0)//2)]
        emb_d2_t = emb_d2[(y_d.size(0)//2):]

        # emb_d1_al = self.layer_d1(emb)
        # emb_d1_al = self.layer_d1_1(emb_d1_al)
        emb_d2_al = self.layer_d(emb)
        emb_d2_al_s = emb_d2_al[:(y_d.size(0)//2)]
        emb_d2_al_t = emb_d2_al[(y_d.size(0)//2):]

        assert L_grad.size(0) == emb_d2_s.size(0) == emb_d2_al_s.size(0) == (y_d.size(0)//2)

        L_wd = emb_d2_s - emb_d2_t
        
        L_wd = - (L_wd.squeeze() - self.gamma * L_grad)
        L_wd = torch.mean(L_wd)

        L_wd_al = emb_d2_al_s - emb_d2_al_t
        L_wd_al = torch.mean(L_wd_al)

        L_grad_back = torch.mean(L_grad.detach())

        return [loss_c, L_wd, L_wd_al], [acc, L_grad_back]
