import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
import torch
import torch.nn as nn
import numpy as np
import models
from torch.autograd import Function
import torch.nn.functional as F


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class Xvector_SAP(torch.nn.Module):
    def __init__(self, feat_dim, emb_dim):
        super(Xvector_SAP, self).__init__()
        self.feature_dim = feat_dim
        self.embedding_dim = emb_dim
        self.in_channels = [self.feature_dim, 512, 512, 512, 512]
        self.layer_sizes = [512, 512, 512, 512, 1500]
        self.kernel_sizes = [5, 3, 3, 1, 1]
        self.dilations = [1,2,3,1,1]
        self.hidden_dim = 3000



        self.tdnn1 = torch.nn.Sequential(
            nn.Conv1d(self.in_channels[0],self.layer_sizes[0],self.kernel_sizes[0],dilation=self.dilations[0]),
            nn.ReLU(True),
            nn.BatchNorm1d(self.layer_sizes[0])
        )
        self.tdnn2 = torch.nn.Sequential(
            nn.Conv1d(self.in_channels[1],self.layer_sizes[1],self.kernel_sizes[1],dilation=self.dilations[1]),
            nn.ReLU(True),
            nn.BatchNorm1d(self.layer_sizes[1])
        )
        self.tdnn3 = torch.nn.Sequential(
            nn.Conv1d(self.in_channels[2],self.layer_sizes[2],self.kernel_sizes[2],dilation=self.dilations[2]),
            nn.ReLU(True),
            nn.BatchNorm1d(self.layer_sizes[2])
        )
        self.tdnn4 = torch.nn.Sequential(
            nn.Conv1d(self.in_channels[3],self.layer_sizes[3],self.kernel_sizes[3],dilation=self.dilations[3]),
            nn.ReLU(True),
            nn.BatchNorm1d(self.layer_sizes[3])
        )
        self.tdnn5 = torch.nn.Sequential(
            nn.Conv1d(self.in_channels[4],self.layer_sizes[4],self.kernel_sizes[4],dilation=self.dilations[4]),
            nn.ReLU(True),
            nn.BatchNorm1d(self.layer_sizes[4])
        )
       
        
        self.embedding_layer1 = torch.nn.Sequential()
        self.embedding_layer1.add_module('linear', nn.Linear(self.hidden_dim, self.embedding_dim))
        self.embedding_layer1.add_module('relu', nn.ReLU(True))
        self.embedding_layer1.add_module('batchnorm',nn.BatchNorm1d(self.embedding_dim))
        
        self.embedding_layer2 = torch.nn.Sequential()
        self.embedding_layer2.add_module('linear', nn.Linear(self.embedding_dim, self.embedding_dim))
        # self.embedding_layer2.add_module('relu', nn.ReLU(True))
        self.embedding_layer2.add_module('batchnorm',nn.BatchNorm1d(self.embedding_dim))


    def stat_pool(self, stat_src):
        stat_mean = torch.mean(stat_src,dim=2)
        stat_std = torch.sqrt(torch.var(stat_src,dim=2)+0.00001)
        stat_pool_out = torch.cat((stat_mean,stat_std),1)
        return stat_pool_out

    def forward(self, x, y):
        src = x.permute(0,2,1)
        out = self.tdnn1(src)
        out = self.tdnn2(out)
        out = self.tdnn3(out)
        out = self.tdnn4(out)
        out = self.tdnn5(out)

        out = self.stat_pool(out)

        out_embedding1 = self.embedding_layer1(out)
        out_embedding2 = self.embedding_layer2(out_embedding1)

        return out_embedding1, out_embedding2

class Xvector_SAP_1L(torch.nn.Module):
    def __init__(self, feat_dim, emb_dim):
        super(Xvector_SAP_1L, self).__init__()
        self.feature_dim = feat_dim
        self.embedding_dim = emb_dim
        self.in_channels = [self.feature_dim, 512, 512, 512, 512]
        self.layer_sizes = [512, 512, 512, 512, 1500]
        self.kernel_sizes = [5, 3, 3, 1, 1]
        self.dilations = [1,2,3,1,1]
        self.hidden_dim = 3000



        self.tdnn1 = torch.nn.Sequential(
            nn.Conv1d(self.in_channels[0],self.layer_sizes[0],self.kernel_sizes[0],dilation=self.dilations[0]),
            nn.ReLU(True),
            nn.BatchNorm1d(self.layer_sizes[0])
        )
        self.tdnn2 = torch.nn.Sequential(
            nn.Conv1d(self.in_channels[1],self.layer_sizes[1],self.kernel_sizes[1],dilation=self.dilations[1]),
            nn.ReLU(True),
            nn.BatchNorm1d(self.layer_sizes[1])
        )
        self.tdnn3 = torch.nn.Sequential(
            nn.Conv1d(self.in_channels[2],self.layer_sizes[2],self.kernel_sizes[2],dilation=self.dilations[2]),
            nn.ReLU(True),
            nn.BatchNorm1d(self.layer_sizes[2])
        )
        self.tdnn4 = torch.nn.Sequential(
            nn.Conv1d(self.in_channels[3],self.layer_sizes[3],self.kernel_sizes[3],dilation=self.dilations[3]),
            nn.ReLU(True),
            nn.BatchNorm1d(self.layer_sizes[3])
        )
        self.tdnn5 = torch.nn.Sequential(
            nn.Conv1d(self.in_channels[4],self.layer_sizes[4],self.kernel_sizes[4],dilation=self.dilations[4]),
            nn.ReLU(True),
            nn.BatchNorm1d(self.layer_sizes[4])
        )
       
        
        self.embedding_layer1 = torch.nn.Sequential()
        self.embedding_layer1.add_module('linear', nn.Linear(self.hidden_dim, self.embedding_dim))
        # self.embedding_layer1.add_module('relu', nn.ReLU(True))
        self.embedding_layer1.add_module('batchnorm',nn.BatchNorm1d(self.embedding_dim))
        


    def stat_pool(self, stat_src):
        stat_mean = torch.mean(stat_src,dim=2)
        stat_std = torch.sqrt(torch.var(stat_src,dim=2)+0.00001)
        stat_pool_out = torch.cat((stat_mean,stat_std),1)
        return stat_pool_out

    def forward(self, x, y):
        src = x.permute(0,2,1)
        out = self.tdnn1(src)
        out = self.tdnn2(out)
        out = self.tdnn3(out)
        out = self.tdnn4(out)
        out = self.tdnn5(out)

        out = self.stat_pool(out)

        out_embedding1 = self.embedding_layer1(out)

        return None, out_embedding1

class AMSoftmax_normfree(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(AMSoftmax_normfree, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label, s, m):
        # print(self.weight[0, 50:60])
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


class AM_normfree_softmax_anneal_ce_head(nn.Module):
    def __init__(self, backbone, model_settings):
        super(AM_normfree_softmax_anneal_ce_head, self).__init__()
        self.th_step = model_settings['anneal_steps']
        self.iter = 0.0
        self.max_m = model_settings['m']
        self.model_settings = model_settings
        
        self.backbone = nn.DataParallel(backbone)
        self.metrics = AMSoftmax_normfree(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], s=model_settings['s'], m=model_settings['m'])
        self.loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, x, y, mod):
        if mod == 'train':
            self.iter += 1.0
            m = min(self.max_m, (self.iter / self.th_step) * self.model_settings['m'])
        else:
            m = 0.0

        emb1, emb2 = self.backbone(x, y)
        logits, nm_W = self.metrics(emb2, y, s=self.model_settings['s'], m=m)
        loss = self.loss(logits, y)

        pred = logits.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        label = y.data.cpu().numpy()
        acc = np.mean((pred == label).astype(int))

        return loss, logits, emb2, acc, 0.0


class Linear_softmax_ce_head(nn.Module):
    def __init__(self, backbone, model_settings):
        super(Linear_softmax_ce_head, self).__init__()
        self.backbone = nn.DataParallel(backbone)
        self.linear = nn.Linear(model_settings['emb_size'], model_settings['class_num'])
        self.loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, x, y, mod):
        emb1, emb2 = self.backbone(x, y)
        logits = self.linear(emb2)
        loss = self.loss(logits, y)

        pred = logits.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        label = y.data.cpu().numpy()        
        acc = np.mean((pred == label).astype(int))

        return loss, logits, emb1, acc, 0.0

class DANN_tester(nn.Module):
    def __init__(self, model_settings):
        super(DANN_tester, self).__init__()
        self.th_step = model_settings['anneal_steps']
        self.iter = 0.0
        self.max_m = model_settings['m']
        self.model_settings = model_settings
        
        self.backbone = nn.DataParallel(Xvector_SAP(model_settings['in_feat'], model_settings['emb_size']))
        self.metrics = AMSoftmax_normfree(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], s=model_settings['s'], m=model_settings['m'])
        self.reverse_l = ReverseLayerF()

        self.layer_d1 = torch.nn.Sequential()
        self.layer_d1.add_module('linear', nn.Linear(model_settings['emb_size'], model_settings['emb_size']))
        self.layer_d1.add_module('relu', nn.ReLU(True))
        self.layer_d1.add_module('batchnorm',nn.BatchNorm1d(model_settings['emb_size']))

        self.layer_d2 = torch.nn.Sequential()
        self.layer_d2.add_module('linear', nn.Linear(model_settings['emb_size'], 2))

        self.loss = torch.nn.CrossEntropyLoss()
    
    def set_totalstep(self, total_step):
        self.total_step = total_step
    
    def forward(self, x, y, mod):
        y_d = (y >= 1211).long()
        if mod == 'train':
            self.iter += 1.0
            m = min(self.max_m, (self.iter / self.th_step) * self.model_settings['m'])
            m_rever = float(self.iter / self.total_step)
            self.alpha_reverse = 2. / (1. + np.exp(-10 * m_rever)) - 1
        else:
            m = 0.0

        emb1, emb2 = self.backbone(x, y)
        logits, nm_W = self.metrics(emb2, y, s=self.model_settings['s'], m=m)
        loss_c = self.loss(logits, y)

        emb_r = self.reverse_l.apply(emb2, self.alpha_reverse)
        emb_d1 = self.layer_d1(emb_r)
        logits_d = self.layer_d2(emb_d1)
        loss_d = self.loss(logits_d, y_d)

        loss = loss_c + loss_d

        if ((self.iter+1) % 50) == 0:
            print(loss_c, loss_d)

        pred = logits.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        label = y.data.cpu().numpy()
        acc = np.mean((pred == label).astype(int))

        return loss, logits, emb2, acc, 0.0

class DANN_tester_3step(nn.Module):
    def __init__(self, model_settings):
        super(DANN_tester_3step, self).__init__()
        self.th_step = model_settings['anneal_steps']
        self.iter = 0.0
        self.max_m = model_settings['m']
        self.model_settings = model_settings
        
        self.backbone = nn.DataParallel(Xvector_SAP(model_settings['in_feat'], model_settings['emb_size']))
        self.metrics = AMSoftmax_normfree(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], s=model_settings['s'], m=model_settings['m'])
        self.reverse_l = ReverseLayerF()

        self.layer_d1 = torch.nn.Sequential()
        self.layer_d1.add_module('linear', nn.Linear(model_settings['emb_size'], model_settings['emb_size']))
        self.layer_d1.add_module('relu', nn.ReLU(True))
        self.layer_d1.add_module('batchnorm',nn.BatchNorm1d(model_settings['emb_size']))

        self.layer_d2 = torch.nn.Sequential()
        self.layer_d2.add_module('linear', nn.Linear(model_settings['emb_size'], 2))

        self.loss = torch.nn.CrossEntropyLoss()
    
    def get_optimizer(self):
        opt_e = torch.optim.SGD(self.backbone.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        opt_c = torch.optim.SGD(self.metrics.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        opt_d = torch.optim.SGD(list(self.layer_d1.parameters())+list(self.layer_d2.parameters()), lr=1e-2, momentum=0.9, weight_decay=5e-4)

        return opt_e, opt_c, opt_d
    
    def set_totalstep(self, total_step):
        self.total_step = total_step
    
    def forward(self, x, y, mod):
        y_d = (y >= 1211).long()
        if mod == 'train':
            self.iter += 1.0
            m = min(self.max_m, (self.iter / self.th_step) * self.model_settings['m'])
            m_rever = float(self.iter / self.total_step)
            self.alpha_reverse = 2. / (1. + np.exp(-10 * m_rever)) - 1
            # self.alpha_reverse = 1.0
        else:
            m = 0.0

        emb1, emb2 = self.backbone(x, y)
        logits, nm_W = self.metrics(emb2, y, s=self.model_settings['s'], m=m)
        loss_c = self.loss(logits, y)

        emb_r = self.reverse_l.apply(emb2, self.alpha_reverse)
        emb_d1 = self.layer_d1(emb_r)
        logits_d = self.layer_d2(emb_d1)
        loss_d = self.loss(logits_d, y_d)

        loss = loss_c + loss_d

        if ((self.iter+1) % 50) == 0:
            print(loss_c.item(), loss_d.item())

        pred = logits.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        label = y.data.cpu().numpy()
        acc = np.mean((pred == label).astype(int))

        pred_d = logits_d.data.cpu().numpy()
        pred_d = np.argmax(pred_d, axis=1)
        label_d = y_d.data.cpu().numpy()
        acc_d = np.mean((pred_d == label_d).astype(int))

        return loss, logits, emb2, acc, acc_d

class DANN_tester_AL(nn.Module):
    def __init__(self, model_settings):
        super(DANN_tester_AL, self).__init__()
        self.th_step = model_settings['anneal_steps']
        self.iter = 0.0
        self.max_m = model_settings['m']
        self.model_settings = model_settings
        
        self.backbone = nn.DataParallel(Xvector_SAP_1L(model_settings['in_feat'], model_settings['emb_size']))
        self.metrics = AMSoftmax_normfree(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], s=model_settings['s'], m=model_settings['m'])
        # self.reverse_l = ReverseLayerF()

        self.layer_d1 = torch.nn.Sequential()
        self.layer_d1.add_module('linear', nn.Linear(model_settings['emb_size'], model_settings['emb_size']))
        self.layer_d1.add_module('relu', nn.ReLU(True))
        self.layer_d1.add_module('batchnorm',nn.BatchNorm1d(model_settings['emb_size']))

        self.layer_d2 = torch.nn.Sequential()
        self.layer_d2.add_module('linear', nn.Linear(model_settings['emb_size'], 2))

        self.loss = torch.nn.CrossEntropyLoss()
    
    def get_optimizer(self):
        opt_e = torch.optim.SGD(self.backbone.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        opt_c = torch.optim.SGD(self.metrics.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        opt_d = torch.optim.SGD(list(self.layer_d1.parameters())+list(self.layer_d2.parameters()), lr=1e-2, momentum=0.9, weight_decay=5e-4)

        return opt_e, opt_c, opt_d
    
    def set_totalstep(self, total_step):
        self.total_step = total_step
    
    def forward(self, x, y, mod):
        y_d = (y >= 1211).long()
        if mod == 'train':
            self.iter += 1.0
            m = min(self.max_m, (self.iter / self.th_step) * self.model_settings['m'])
        else:
            m = 0.0

        emb1, emb2 = self.backbone(x, y)
        logits, nm_W = self.metrics(emb2, y, s=self.model_settings['s'], m=m)
        loss_c = self.loss(logits, y)

        # emb_r = self.reverse_l.apply(emb2, self.alpha_reverse)
        emb_d1 = self.layer_d1(emb2)
        logits_d = self.layer_d2(emb_d1)
        loss_d = self.loss(logits_d, y_d)

        # loss_al = self.loss(logits_d, torch.zeros(y_d.size()).long().cuda())
        loss_al = self.loss(logits_d, 1-y_d)

        if ((self.iter+1) % 50) == 0:
            print(loss_c.item(), loss_d.item())

        pred = logits.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        label = y.data.cpu().numpy()
        acc = np.mean((pred == label).astype(int))

        pred_d = logits_d.data.cpu().numpy()
        pred_d = np.argmax(pred_d, axis=1)
        label_d = y_d.data.cpu().numpy()
        acc_d = np.mean((pred_d == label_d).astype(int))

        return [loss_c, loss_d, loss_al], logits, emb2, acc, acc_d

class DANN_tester_AL_w(nn.Module):
    def __init__(self, model_settings):
        super(DANN_tester_AL_w, self).__init__()
        self.th_step = model_settings['anneal_steps']
        self.iter = 0.0
        self.max_m = model_settings['m']
        self.model_settings = model_settings
        
        self.backbone = nn.DataParallel(Xvector_SAP_1L(model_settings['in_feat'], model_settings['emb_size']))
        self.metrics = AMSoftmax_normfree(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], s=model_settings['s'], m=model_settings['m'])
        # self.reverse_l = ReverseLayerF()

        self.layer_d1 = torch.nn.Sequential()
        self.layer_d1.add_module('linear', nn.Linear(model_settings['emb_size'], model_settings['emb_size']))
        self.layer_d1.add_module('relu', nn.ReLU(True))
        self.layer_d1.add_module('batchnorm',nn.BatchNorm1d(model_settings['emb_size']))

        self.layer_d2 = torch.nn.Sequential()
        self.layer_d2.add_module('linear', nn.Linear(model_settings['emb_size'], 2))

        self.loss = torch.nn.CrossEntropyLoss(reduction='none')
    
    def get_optimizer(self):
        opt_e = torch.optim.SGD(self.backbone.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        opt_c = torch.optim.SGD(self.metrics.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        opt_d = torch.optim.SGD(list(self.layer_d1.parameters())+list(self.layer_d2.parameters()), lr=1e-2, momentum=0.9, weight_decay=5e-4)

        return opt_e, opt_c, opt_d
    
    def set_totalstep(self, total_step):
        self.total_step = total_step
    
    def forward(self, x, y, mod):
        y_d = (y >= 1211).long()
        weight = y_d*(self.model_settings['weight']-1) + 1

        if mod == 'train':
            self.iter += 1.0
            m = min(self.max_m, (self.iter / self.th_step) * self.model_settings['m'])
        else:
            m = 0.0

        emb1, emb2 = self.backbone(x, y)
        logits, nm_W = self.metrics(emb2, y, s=self.model_settings['s'], m=m)
        loss_c = self.loss(logits, y)
        loss_c = torch.mean(loss_c)

        # emb_r = self.reverse_l.apply(emb2, self.alpha_reverse)
        emb_d1 = self.layer_d1(emb2)
        logits_d = self.layer_d2(emb_d1)
        loss_d = self.loss(logits_d, y_d) * weight
        loss_d = torch.mean(loss_d)

        # loss_al = self.loss(logits_d, torch.zeros(y_d.size()).long().cuda())
        loss_al = self.loss(logits_d, 1-y_d) * weight
        loss_al = torch.mean(loss_al)

        if ((self.iter+1) % 50) == 0:
            print(loss_c.item(), loss_d.item())

        pred = logits.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        label = y.data.cpu().numpy()
        acc = np.mean((pred == label).astype(int))

        pred_d = logits_d.data.cpu().numpy()
        pred_d = np.argmax(pred_d, axis=1)
        label_d = y_d.data.cpu().numpy()
        acc_d = np.mean((pred_d == label_d).astype(int))

        return [loss_c, loss_d, loss_al], logits, emb2, acc, acc_d

class DANN_tester_AL_w_changeadv(nn.Module):
    def __init__(self, model_settings):
        super(DANN_tester_AL_w_changeadv, self).__init__()
        self.th_step = model_settings['anneal_steps']
        self.iter = 0.0
        self.max_m = model_settings['m']
        self.model_settings = model_settings
        
        self.backbone = nn.DataParallel(Xvector_SAP_1L(model_settings['in_feat'], model_settings['emb_size']))
        self.metrics = AMSoftmax_normfree(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], s=model_settings['s'], m=model_settings['m'])
        # self.reverse_l = ReverseLayerF()

        self.layer_d1 = torch.nn.Sequential()
        self.layer_d1.add_module('linear', nn.Linear(model_settings['emb_size'], model_settings['emb_size']))
        self.layer_d1.add_module('relu', nn.ReLU(True))
        self.layer_d1.add_module('batchnorm',nn.BatchNorm1d(model_settings['emb_size']))

        self.layer_d2 = torch.nn.Sequential()
        self.layer_d2.add_module('linear', nn.Linear(model_settings['emb_size'], 2))

        self.loss = torch.nn.CrossEntropyLoss(reduction='none')

        self.logsft = torch.nn.LogSoftmax(dim=1)
        # self.negloss = torch.nn.NLLLoss()
    
    def get_optimizer(self):
        opt_e = torch.optim.SGD(self.backbone.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        opt_c = torch.optim.SGD(self.metrics.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        opt_d = torch.optim.SGD(list(self.layer_d1.parameters())+list(self.layer_d2.parameters()), lr=1e-2, momentum=0.9, weight_decay=5e-4)

        return opt_e, opt_c, opt_d
    
    def set_totalstep(self, total_step):
        self.total_step = total_step
    
    def forward(self, x, y, mod):
        y_d = (y >= 1211).long()
        weight = (y_d*(self.model_settings['weight']-1) + 1).float()

        if mod == 'train':
            self.iter += 1.0
            m = min(self.max_m, (self.iter / self.th_step) * self.model_settings['m'])
        else:
            m = 0.0

        emb1, emb2 = self.backbone(x, y)
        logits, nm_W = self.metrics(emb2, y, s=self.model_settings['s'], m=m)
        loss_c = self.loss(logits, y)
        loss_c = torch.mean(loss_c)

        # emb_r = self.reverse_l.apply(emb2, self.alpha_reverse)
        emb_d1 = self.layer_d1(emb2)
        logits_d = self.layer_d2(emb_d1)
        loss_d = self.loss(logits_d, y_d) * weight
        loss_d = torch.mean(loss_d)

        # loss_al = self.loss(logits_d, torch.zeros(y_d.size()).long().cuda())
        # loss_al = self.loss(logits_d, 1-y_d) * weight
        # loss_al = torch.mean(loss_al)

        loss_al = -0.5 * self.logsft(logits_d)
        loss_al = loss_al * weight.unsqueeze(1)
        loss_al = torch.mean(loss_al)

        # if ((self.iter+1) % 50) == 0:
        #     print(loss_c.item(), loss_d.item())

        pred = logits.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        label = y.data.cpu().numpy()
        acc = np.mean((pred == label).astype(int))

        pred_d = logits_d.data.cpu().numpy()
        pred_d = np.argmax(pred_d, axis=1)
        label_d = y_d.data.cpu().numpy()
        acc_d = np.mean((pred_d == label_d).astype(int))

        return [loss_c, loss_d, loss_al], logits, emb2, [acc, acc_d], None

class DANN_tester_AL_w_changeadv_full(nn.Module):
    def __init__(self, model_settings):
        super(DANN_tester_AL_w_changeadv_full, self).__init__()
        self.th_step = model_settings['anneal_steps']
        self.iter = 0.0
        self.max_m = model_settings['m']
        self.model_settings = model_settings
        
        self.backbone = nn.DataParallel(Xvector_SAP_1L(model_settings['in_feat'], model_settings['emb_size']))
        self.metrics = AMSoftmax_normfree(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], s=model_settings['s'], m=model_settings['m'])
        # self.reverse_l = ReverseLayerF()

        self.layer_d1 = torch.nn.Sequential()
        self.layer_d1.add_module('linear', nn.Linear(model_settings['emb_size'], model_settings['emb_size']))
        self.layer_d1.add_module('relu', nn.ReLU(True))
        self.layer_d1.add_module('batchnorm',nn.BatchNorm1d(model_settings['emb_size']))

        self.layer_d2 = torch.nn.Sequential()
        self.layer_d2.add_module('linear', nn.Linear(model_settings['emb_size'], 2))

        self.loss = torch.nn.CrossEntropyLoss(reduction='none')

        self.logsft = torch.nn.LogSoftmax(dim=1)
        # self.negloss = torch.nn.NLLLoss()
    
    def get_optimizer(self):
        opt_e = torch.optim.SGD(self.backbone.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        opt_c = torch.optim.SGD(self.metrics.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        opt_d = torch.optim.SGD(list(self.layer_d1.parameters())+list(self.layer_d2.parameters()), lr=1e-2, momentum=0.9, weight_decay=5e-4)  

        return opt_e, opt_c, opt_d
    
    def set_totalstep(self, total_step):
        self.total_step = total_step
    
    def forward(self, x, y, mod):
        y_d = (y >= 7323).long()
        weight = (y_d*(self.model_settings['weight']-1) + 1).float()

        if mod == 'train':
            self.iter += 1.0
            m = min(self.max_m, (self.iter / self.th_step) * self.model_settings['m'])
        else:
            m = 0.0

        emb1, emb2 = self.backbone(x, y)
        logits, nm_W = self.metrics(emb2, y, s=self.model_settings['s'], m=m)
        loss_c = self.loss(logits, y)
        loss_c = torch.mean(loss_c)

        # emb_r = self.reverse_l.apply(emb2, self.alpha_reverse)
        emb_d1 = self.layer_d1(emb2)
        logits_d = self.layer_d2(emb_d1)
        loss_d = self.loss(logits_d, y_d) * weight
        loss_d = torch.mean(loss_d)

        # loss_al = self.loss(logits_d, torch.zeros(y_d.size()).long().cuda())
        # loss_al = self.loss(logits_d, 1-y_d) * weight
        # loss_al = torch.mean(loss_al)

        loss_al = -0.5 * self.logsft(logits_d)
        loss_al = loss_al * weight.unsqueeze(1)
        loss_al = torch.mean(loss_al)

        # if ((self.iter+1) % 50) == 0:
        #     print(loss_c.item(), loss_d.item())

        pred = logits.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        label = y.data.cpu().numpy()
        acc = np.mean((pred == label).astype(int))

        pred_d = logits_d.data.cpu().numpy()
        pred_d = np.argmax(pred_d, axis=1)
        label_d = y_d.data.cpu().numpy()
        acc_d = np.mean((pred_d == label_d).astype(int))

        return [loss_c, loss_d, loss_al], logits, emb2, [acc, acc_d], None

class DANN_tester_AL_FOCAL(nn.Module):
    def __init__(self, model_settings):
        super(DANN_tester_AL_FOCAL, self).__init__()
        self.th_step = model_settings['anneal_steps']
        self.iter = 0.0
        self.max_m = model_settings['m']
        self.model_settings = model_settings
        
        self.backbone = nn.DataParallel(Xvector_SAP_1L(model_settings['in_feat'], model_settings['emb_size']))
        self.metrics = AMSoftmax_normfree(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], s=model_settings['s'], m=model_settings['m'])
        # self.reverse_l = ReverseLayerF()

        self.layer_d1 = torch.nn.Sequential()
        self.layer_d1.add_module('linear', nn.Linear(model_settings['emb_size'], model_settings['emb_size']))
        self.layer_d1.add_module('relu', nn.ReLU(True))
        self.layer_d1.add_module('batchnorm',nn.BatchNorm1d(model_settings['emb_size']))

        self.layer_d2 = torch.nn.Sequential()
        self.layer_d2.add_module('linear', nn.Linear(model_settings['emb_size'], 2))

        self.loss = torch.nn.CrossEntropyLoss(reduction='none')

        self.sft = torch.nn.Softmax(dim=1)
        self.negloss = torch.nn.NLLLoss()
    
    def get_optimizer(self):
        opt_e = torch.optim.SGD(self.backbone.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        opt_c = torch.optim.SGD(self.metrics.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        opt_d = torch.optim.SGD(list(self.layer_d1.parameters())+list(self.layer_d2.parameters()), lr=1e-2, momentum=0.9, weight_decay=5e-4)

        return opt_e, opt_c, opt_d
    
    def set_totalstep(self, total_step):
        self.total_step = total_step
    
    def forward(self, x, y, mod):
        y_d = (y >= 1211).long()
        weight = (y_d*(self.model_settings['weight']-1) + 1).unsqueeze(1).float()

        if mod == 'train':
            self.iter += 1.0
            m = min(self.max_m, (self.iter / self.th_step) * self.model_settings['m'])
        else:
            m = 0.0

        emb1, emb2 = self.backbone(x, y)
        logits, nm_W = self.metrics(emb2, y, s=self.model_settings['s'], m=m)
        loss_c = self.loss(logits, y)
        loss_c = torch.mean(loss_c)

        # emb_r = self.reverse_l.apply(emb2, self.alpha_reverse)
        emb_d1 = self.layer_d1(emb2)
        logits_d = self.layer_d2(emb_d1)

        loss_d = self.negloss(weight * ((1-self.sft(logits_d)) ** self.model_settings['focal_d_gamma']) * torch.log(self.sft(logits_d)), y_d)
        loss_d = torch.mean(loss_d)


        loss_al = torch.sum((-1) * weight * ((1-self.sft(logits_d)) ** self.model_settings['focal_al_gamma']) * torch.log(self.sft(logits_d)), dim=1)
        loss_al = torch.mean(loss_al)


        pred = logits.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        label = y.data.cpu().numpy()
        acc = np.mean((pred == label).astype(int))

        pred_d = logits_d.data.cpu().numpy()
        pred_d = np.argmax(pred_d, axis=1)
        label_d = y_d.data.cpu().numpy()
        acc_d = np.mean((pred_d == label_d).astype(int))

        return [loss_c, loss_d, loss_al], logits, emb2, [acc, acc_d], None

class DANN_tester_AL_FOCAL_REBL(nn.Module):
    def __init__(self, model_settings):
        super(DANN_tester_AL_FOCAL_REBL, self).__init__()
        self.th_step = model_settings['anneal_steps']
        self.iter = 0.0
        self.max_m = model_settings['m']
        self.model_settings = model_settings
        
        self.backbone = nn.DataParallel(Xvector_SAP_1L(model_settings['in_feat'], model_settings['emb_size']))
        self.metrics = AMSoftmax_normfree(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], s=model_settings['s'], m=model_settings['m'])
        # self.reverse_l = ReverseLayerF()

        self.layer_d1 = torch.nn.Sequential()
        self.layer_d1.add_module('linear', nn.Linear(model_settings['emb_size'], model_settings['emb_size']))
        self.layer_d1.add_module('relu', nn.ReLU(True))
        self.layer_d1.add_module('batchnorm',nn.BatchNorm1d(model_settings['emb_size']))

        self.layer_d2 = torch.nn.Sequential()
        self.layer_d2.add_module('linear', nn.Linear(model_settings['emb_size'], 2))

        self.loss = torch.nn.CrossEntropyLoss(reduction='none')

        self.sft = torch.nn.Softmax(dim=1)
        self.negloss = torch.nn.NLLLoss(reduction='none')
    
    def get_optimizer(self):
        opt_e = torch.optim.SGD(self.backbone.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        opt_c = torch.optim.SGD(self.metrics.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        opt_d = torch.optim.SGD(list(self.layer_d1.parameters())+list(self.layer_d2.parameters()), lr=1e-2, momentum=0.9, weight_decay=5e-4)

        return opt_e, opt_c, opt_d
    
    def set_totalstep(self, total_step):
        self.total_step = total_step
    
    def forward(self, x, y, mod):
        y_d = (y >= 1211).long()
        weight = (y_d*(self.model_settings['weight']-1) + 1).unsqueeze(1).float()
        reblanced_batch_size = y.size(0) * 2 * (self.model_settings['weight']) / (self.model_settings['weight']+1)

        if mod == 'train':
            self.iter += 1.0
            m = min(self.max_m, (self.iter / self.th_step) * self.model_settings['m'])
        else:
            m = 0.0

        emb1, emb2 = self.backbone(x, y)
        logits, nm_W = self.metrics(emb2, y, s=self.model_settings['s'], m=m)
        loss_c = self.loss(logits, y)
        loss_c = torch.mean(loss_c)

        # emb_r = self.reverse_l.apply(emb2, self.alpha_reverse)
        emb_d1 = self.layer_d1(emb2)
        logits_d = self.layer_d2(emb_d1)

        loss_d = self.negloss(weight * ((1-self.sft(logits_d)) ** self.model_settings['focal_d_gamma']) * torch.log(self.sft(logits_d)), y_d)
        loss_d = torch.sum(loss_d) / reblanced_batch_size


        loss_al = torch.sum((-1) * weight * ((1-self.sft(logits_d)) ** self.model_settings['focal_al_gamma']) * torch.log(self.sft(logits_d)), dim=1)
        loss_al = torch.sum(loss_al) / reblanced_batch_size


        pred = logits.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        label = y.data.cpu().numpy()
        acc = np.mean((pred == label).astype(int))

        pred_d = logits_d.data.cpu().numpy()
        pred_d = np.argmax(pred_d, axis=1)
        label_d = y_d.data.cpu().numpy()
        acc_d = np.mean((pred_d == label_d).astype(int))

        return [loss_c, loss_d, loss_al], logits, emb2, [acc, acc_d], None

class DANN_tester_AL_w_SAP(nn.Module):
    def __init__(self, model_settings):
        super(DANN_tester_AL_w_SAP, self).__init__()
        self.th_step = model_settings['anneal_steps']
        self.iter = 0.0
        self.max_m = model_settings['m']
        self.model_settings = model_settings
        
        self.backbone = nn.DataParallel(Xvector_SAP(model_settings['in_feat'], model_settings['emb_size']))
        self.metrics = AMSoftmax_normfree(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], s=model_settings['s'], m=model_settings['m'])
        # self.reverse_l = ReverseLayerF()

        self.layer_d1 = torch.nn.Sequential()
        self.layer_d1.add_module('linear', nn.Linear(model_settings['emb_size'], model_settings['emb_size']))
        self.layer_d1.add_module('relu', nn.ReLU(True))
        self.layer_d1.add_module('batchnorm',nn.BatchNorm1d(model_settings['emb_size']))

        self.layer_d2 = torch.nn.Sequential()
        self.layer_d2.add_module('linear', nn.Linear(model_settings['emb_size'], 2))

        self.loss = torch.nn.CrossEntropyLoss(reduction='none')
    
    def get_optimizer(self):
        opt_e = torch.optim.SGD(self.backbone.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        opt_c = torch.optim.SGD(self.metrics.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        opt_d = torch.optim.SGD(list(self.layer_d1.parameters())+list(self.layer_d2.parameters()), lr=1e-2, momentum=0.9, weight_decay=5e-4)

        return opt_e, opt_c, opt_d
    
    def set_totalstep(self, total_step):
        self.total_step = total_step
    
    def forward(self, x, y, mod):
        y_d = (y >= 1211).long()
        weight = y_d*(self.model_settings['weight']-1) + 1

        if mod == 'train':
            self.iter += 1.0
            m = min(self.max_m, (self.iter / self.th_step) * self.model_settings['m'])
        else:
            m = 0.0

        emb1, emb2 = self.backbone(x, y)
        logits, nm_W = self.metrics(emb2, y, s=self.model_settings['s'], m=m)
        loss_c = self.loss(logits, y)
        loss_c = torch.mean(loss_c)

        # emb_r = self.reverse_l.apply(emb2, self.alpha_reverse)
        emb_d1 = self.layer_d1(emb2)
        logits_d = self.layer_d2(emb_d1)
        loss_d = self.loss(logits_d, y_d) * weight
        loss_d = torch.mean(loss_d)

        # loss_al = self.loss(logits_d, torch.zeros(y_d.size()).long().cuda())
        loss_al = self.loss(logits_d, 1-y_d) * weight
        loss_al = torch.mean(loss_al)

        if ((self.iter+1) % 50) == 0:
            print(loss_c.item(), loss_d.item())

        pred = logits.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        label = y.data.cpu().numpy()
        acc = np.mean((pred == label).astype(int))

        pred_d = logits_d.data.cpu().numpy()
        pred_d = np.argmax(pred_d, axis=1)
        label_d = y_d.data.cpu().numpy()
        acc_d = np.mean((pred_d == label_d).astype(int))

        return [loss_c, loss_d, loss_al], logits, emb2, acc, acc_d

class GAN_tester(nn.Module):
    def __init__(self, model_settings):
        super(GAN_tester, self).__init__()
        self.th_step = model_settings['anneal_steps']
        self.iter = 0.0
        self.max_m = model_settings['m']
        self.model_settings = model_settings
        
        self.backbone = nn.DataParallel(Xvector_SAP(model_settings['in_feat'], model_settings['emb_size']))
        self.metrics = AMSoftmax_normfree(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], s=model_settings['s'], m=model_settings['m'])
        # self.reverse_l = ReverseLayerF()

        self.layer_d1 = torch.nn.Sequential()
        self.layer_d1.add_module('linear', nn.Linear(model_settings['emb_size'], model_settings['emb_size']))
        self.layer_d1.add_module('relu', nn.ReLU(True))
        self.layer_d1.add_module('batchnorm',nn.BatchNorm1d(model_settings['emb_size']))

        self.layer_d2 = torch.nn.Sequential()
        self.layer_d2.add_module('linear', nn.Linear(model_settings['emb_size'], 1))

        self.loss = torch.nn.CrossEntropyLoss()

        self.bce_loss = torch.nn.BCEWithLogitsLoss()

        self.get_optimizer()
    
    def get_optimizer(self):
        opt_e = torch.optim.SGD(self.backbone.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        opt_c = torch.optim.SGD(self.metrics.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        opt_d = torch.optim.SGD(list(self.layer_d1.parameters())+list(self.layer_d2.parameters()), lr=1e-2, momentum=0.9, weight_decay=5e-4)

        # opt_e = torch.optim.Adam(self.backbone.parameters(), lr=3e-3)
        # opt_c = torch.optim.Adam(self.metrics.parameters(), lr=3e-3)
        # opt_d = torch.optim.Adam(list(self.layer_d1.parameters())+list(self.layer_d2.parameters()), lr=3e-3)

        return opt_e, opt_c, opt_d
    
    def set_totalstep(self, total_step):
        self.total_step = total_step
    
    # def forward(self, x_s, y_s, x_t, y_t, mod):
    #     y_d = torch.cat([torch.ones(y_s.size(0)), torch.zeros(y_t.size(0))], axis=0).cuda()
    #     if mod == 'train':
    #         self.iter += 1.0
    #         m = min(self.max_m, (self.iter / self.th_step) * self.model_settings['m'])
    #     else:
    #         m = 0.0

    #     emb1, emb2 = self.backbone(torch.cat([x_s, x_t], axis=0), torch.cat([y_s, y_t], axis=0))
    #     logits, nm_W = self.metrics(emb2, torch.cat([y_s, y_t]), s=self.model_settings['s'], m=m)
    #     loss_c = self.loss(logits, torch.cat([y_s, y_t]))

    #     # emb_r = self.reverse_l.apply(emb2, self.alpha_reverse)
    #     emb_d1 = self.layer_d1(emb2)
    #     logits_d = self.layer_d2(emb_d1)
    #     loss_d = self.bce_loss(logits_d, y_d[:,None])
    #     # print(logits_d[-y_t.size(0):])

    #     loss_d_adv = self.bce_loss(logits_d[-y_t.size(0):], torch.ones(y_t.size(0)).cuda()[:, None])

    #     # loss_al = self.loss(logits_d, torch.zeros(y_d.size()).long().cuda())
    #     # loss_al = self.loss(logits_d, 1-y_d)

    #     # loss = loss_c + loss_d

    #     if ((self.iter+1) % 50) == 0:
    #         print(loss_c.item(), loss_d.item())        

    #     pred = logits.data.cpu().numpy()
    #     pred = np.argmax(pred, axis=1)
    #     label = torch.cat([y_s, y_t]).data.cpu().numpy()
    #     acc = np.mean((pred == label).astype(int))

    #     pred_d = logits_d.data.cpu().numpy()
    #     pred_d = np.concatenate([1-pred_d, pred_d], axis=1)
    #     pred_d = np.argmax(pred_d, axis=1)
    #     label_d = y_d.data.cpu().numpy()
    #     acc_d = np.mean((pred_d == label_d).astype(int))

    #     return [loss_c, loss_d, loss_d_adv], logits, emb2, acc, acc_d

    def forward(self, x, y, mod):       
        if mod == 'train':
            self.iter += 1.0
            m = min(self.max_m, (self.iter / self.th_step) * self.model_settings['m'])
            y_d = torch.cat([torch.ones(y.size(0) // 2), torch.zeros(y.size(0) // 2)], axis=0).cuda()
        else:
            m = 0.0
            y_d = (y < 1211).float().cuda()

        emb1, emb2 = self.backbone(x, y)
        logits, nm_W = self.metrics(emb2, y, s=self.model_settings['s'], m=m)
        loss_c = self.loss(logits, y)

        # emb_r = self.reverse_l.apply(emb2, self.alpha_reverse)
        emb_d1 = self.layer_d1(emb2)
        logits_d = self.layer_d2(emb_d1)
        loss_d = self.bce_loss(logits_d, y_d[:,None])
        # print(logits_d[-y_t.size(0):])

        if mod == 'train':
            loss_d_adv = self.bce_loss(logits_d[-y.size(0)//2:], torch.ones(y.size(0)//2).cuda()[:, None])
        else:
            loss_d_adv = None

        # loss_al = self.loss(logits_d, torch.zeros(y_d.size()).long().cuda())
        # loss_al = self.loss(logits_d, 1-y_d)

        # loss = loss_c + loss_d

        if ((self.iter+1) % 50) == 0:
            print(loss_c.item(), loss_d.item())        

        pred = logits.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        label = y.data.cpu().numpy()
        acc = np.mean((pred == label).astype(int))

        pred_d = logits_d.data.cpu().numpy()
        pred_d = np.concatenate([-pred_d, pred_d], axis=1)
        pred_d = np.argmax(pred_d, axis=1)
        label_d = y_d.data.cpu().numpy()
        acc_d = np.mean((pred_d == label_d).astype(int))

        return [loss_c, loss_d, loss_d_adv], logits, emb2, acc, acc_d

