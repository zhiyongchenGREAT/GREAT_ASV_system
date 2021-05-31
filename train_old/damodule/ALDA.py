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

__all__ = ["FOCAL_ALDA_MULDO_OPT_FAST"]

class FOCAL_ALDA_MULDO_OPT_FAST(nn.Module):
    def __init__(self, spk_clf_head, spk_backbone, nOut, domain_classes, ori_weight_dict, **kwargs):
        super(FOCAL_ALDA_MULDO_OPT_FAST, self).__init__()
        # self.th_step = model_settings['anneal_steps']
        # self.iter = 0.0
        # self.max_m = model_settings['m']
        self.emb_size = nOut
        self.domain_classes = domain_classes
        
        # self.backbone = nn.DataParallel(Xvector_SAP_1L(model_settings['in_feat'], model_settings['emb_size']))
        self.backbone = spk_backbone
        # self.metrics = AMSoftmax_normfree(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], s=model_settings['s'], m=model_settings['m'])
        self.metrics = spk_clf_head

        self.layer_d1 = torch.nn.Sequential()
        self.layer_d1.add_module('linear', nn.Linear(self.emb_size, self.emb_size))
        self.layer_d1.add_module('relu', nn.ReLU(True))
        self.layer_d1.add_module('batchnorm',nn.BatchNorm1d(self.emb_size))

        self.layer_d2 = torch.nn.Sequential()
        self.layer_d2.add_module('linear', nn.Linear(self.emb_size, self.domain_classes))

        self.loss = torch.nn.CrossEntropyLoss(reduction='none')

        self.sft = torch.nn.Softmax(dim=1)
        self.negloss = torch.nn.NLLLoss(reduction='none')

        assert len(ori_weight_dict) == domain_classes

        ori_weight_dict2 = {}
        for i in list(ori_weight_dict.keys()):
            ori_weight_dict2[int(i)] = int(ori_weight_dict[i])

        total_weight = 0
        for i in ori_weight_dict2:
            total_weight += ori_weight_dict2[i]
        # weight_split = np.array(self.model_settings['weight_split'])
        self.weight_dict = {}
        for i in ori_weight_dict2:
            self.weight_dict[i] = total_weight / (len(ori_weight_dict2)*ori_weight_dict2[i])
    
    def get_optimizer(self, optimizer, **kwargs):
        # opt_e_c = torch.optim.SGD(list(self.backbone.parameters())+list(self.metrics.parameters()), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        # opt_c = torch.optim.SGD(self.metrics.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        # opt_d = torch.optim.SGD(list(self.layer_d1.parameters())+list(self.layer_d2.parameters()), lr=1e-2, momentum=0.9, weight_decay=5e-4)

        Optimizer = importlib.import_module('optimizer.'+optimizer).__getattribute__('Optimizer')
        self.opt_e_c = Optimizer(list(self.backbone.parameters())+list(self.metrics.parameters()), **kwargs)
        self.opt_d = Optimizer(list(self.layer_d1.parameters())+list(self.layer_d2.parameters()), **kwargs)

        # print(self.opt_e_c)
        # print(self.opt_d)

        return self.opt_e_c, self.opt_d

    def get_scheduler(self, scheduler, **kwargs):
        Scheduler = importlib.import_module('scheduler.'+scheduler).__getattribute__('Scheduler')
        self.sche_e_c, self.lr_step, self.expected_step = Scheduler(self.opt_e_c, **kwargs)
        self.sche_d, _, _ = Scheduler(self.opt_d, **kwargs)

        return self.sche_e_c, self.sche_d, self.lr_step, self.expected_step


    
    def forward(self, emb, y, y_d):
        weight = torch.zeros(y_d.size(), dtype=torch.float).cuda()
        # y_d = torch.zeros(y.size(), dtype=torch.long).cuda()

        # for i in range(len(self.model_settings['class_split'])-1):
        #     weight[(y >= self.model_settings['class_split'][i]) & (y < self.model_settings['class_split'][i+1])] = self.weight_list[i]
        #     y_d[(y >= self.model_settings['class_split'][i]) & (y < self.model_settings['class_split'][i+1])] = i
        for count, i in enumerate(y_d):
            weight[count] = self.weight_dict[int(i.data)]

        weight = weight.unsqueeze(1)

        # print(weight)

        # if mod == 'train':
        #     self.iter += 1.0
        #     m = min(self.max_m, (self.iter / self.th_step) * self.model_settings['m'])
        # else:
        #     m = 0.0

        # emb1, emb2 = self.backbone(x, y)
        # logits, nm_W = self.metrics(emb2, y, s=self.model_settings['s'], m=m)
        # loss_c = self.loss(logits, y)
        # loss_c = torch.mean(loss_c)
        loss_c, acc = self.metrics(emb, y)

        emb_d1 = self.layer_d1(emb.detach())
        logits_d = self.layer_d2(emb_d1)

        emb_d1_al = self.layer_d1(emb)
        logits_d_al = self.layer_d2(emb_d1_al)

        loss_d = self.negloss(weight * torch.log(self.sft(logits_d)), y_d)
        loss_d = torch.mean(loss_d)

        loss_al = torch.mean((-1) * weight * torch.log(self.sft(logits_d_al)), dim=1)
        loss_al = torch.mean(loss_al)

        # pred = logits.data.cpu().numpy()
        # pred = np.argmax(pred, axis=1)
        # label = y.data.cpu().numpy()
        # acc = np.mean((pred == label).astype(int))

        pred_d = logits_d.data.cpu().numpy()
        pred_d = np.argmax(pred_d, axis=1)
        label_d = y_d.data.cpu().numpy()
        acc_d = np.mean((pred_d == label_d).astype(int))

        return [loss_c, loss_d, loss_al], [acc, acc_d]
