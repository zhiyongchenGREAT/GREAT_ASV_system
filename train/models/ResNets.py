import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.metrics import *


class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes/2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)
    
    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class PreactivateBlock(nn.Module):
    def __init__(self, input_dim, output_dim, downsample=False, firstact=True):
        super(PreactivateBlock, self).__init__()
        self.downsample = downsample
        self.firstact = firstact
        self.bn1 = nn.BatchNorm2d(input_dim)
        self.relu1 = nn.ReLU()
        if downsample is True:
            self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=2, padding=1)
        elif downsample is False:
            self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1)
        else:
            raise Exception('downsample wrong setting')

        self.bn2 = nn.BatchNorm2d(output_dim)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=output_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1)
        
        self.downsampler = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=1, stride=2, padding=0)
        )

    def forward(self, x):
        residual = x
        if self.firstact:
            out = self.bn1(x)
            out = self.relu1(out)
            downsample_out = out
            out = self.conv1(out)
        else:
            out = self.conv1(x)

        out = self.bn2(out)
        out = self.relu2(out)        
        out = self.conv2(out)

        if self.downsample is True:
            residual = self.downsampler(downsample_out)

        out += residual
        return out

class PreactivateMaxpoolBlock(nn.Module):
    def __init__(self, input_dim, output_dim, downsample=False, firstact=True):
        super(PreactivateMaxpoolBlock, self).__init__()
        self.downsample = downsample
        self.firstact = firstact
        self.bn1 = nn.BatchNorm2d(input_dim)
        self.relu1 = nn.ReLU()
        if downsample is True:
            self.conv1 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=1, stride=1, padding=0)
            )

        elif downsample is False:
            self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1)
        else:
            raise Exception('downsample wrong setting')

        self.bn2 = nn.BatchNorm2d(output_dim)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=output_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1)
        
        self.downsampler = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        residual = x
        if self.firstact:
            out = self.bn1(x)
            out = self.relu1(out)
            downsample_out = out
            out = self.conv1(out)
        else:
            out = self.conv1(x)

        out = self.bn2(out)
        out = self.relu2(out)        
        out = self.conv2(out)

        if self.downsample is True:
            residual = self.downsampler(downsample_out)

        out += residual
        return out

class IBNPreactivateBlock(nn.Module):
    def __init__(self, input_dim, output_dim, downsample=False, firstact=True):
        super(IBNPreactivateBlock, self).__init__()
        self.downsample = downsample
        self.firstact = firstact
        self.bn1 = IBN(input_dim)
        self.relu1 = nn.ReLU()
        if downsample is True:
            self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=2, padding=1)
        elif downsample is False:
            self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1)
        else:
            raise Exception('downsample wrong setting')

        self.bn2 = IBN(output_dim)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=output_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1)
        
        self.downsampler = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=1, stride=2, padding=0)
        )

    def forward(self, x):
        residual = x
        if self.firstact:
            out = self.bn1(x)
            out = self.relu1(out)
            downsample_out = out
            out = self.conv1(out)
        else:
            out = self.conv1(x)

        out = self.bn2(out)
        out = self.relu2(out)        
        out = self.conv2(out)

        if self.downsample is True:
            residual = self.downsampler(downsample_out)

        out += residual
        return out


class SePreactivateBlock(nn.Module):
    def __init__(self, input_dim, output_dim, downsample=False, firstact=True):
        super(SePreactivateBlock, self).__init__()
        self.downsample = downsample
        self.firstact = firstact
        self.bn1 = nn.BatchNorm2d(input_dim)
        self.relu1 = nn.ReLU()
        if downsample is True:
            self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=2, padding=1)
        elif downsample is False:
            self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1)
        else:
            raise Exception('downsample wrong setting')

        self.bn2 = nn.BatchNorm2d(output_dim)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=output_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1)
        
        self.downsampler = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=1, stride=2, padding=0)
        )
        
        # SE block
        self.globalAvgPool = nn.AdaptiveAvgPool2d([1, 1])
        self.SE_Block = nn.Sequential(
            nn.Linear(output_dim, int(output_dim/16)),
            nn.ReLU(),
            nn.Linear(int(output_dim/16), output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        if self.firstact:
            out = self.bn1(x)
            out = self.relu1(out)
            downsample_out = out
            out = self.conv1(out)
        else:
            out = self.conv1(x)

        out = self.bn2(out)
        out = self.relu2(out)        
        out = self.conv2(out)

        if self.downsample is True:
            residual = self.downsampler(downsample_out)

        original_out = out
        out = self.globalAvgPool(out)
        out = out.view(out.shape[0], -1)
        out = self.SE_Block(out)
        out = out.view(out.shape[0], out.shape[1], 1, 1)
        out = original_out * out

        out += residual
        return out


class Resnet18(nn.Module):
    def __init__(self, feat_dim, emb_dim):
        super(Resnet18, self).__init__()
        self.emb_dim = emb_dim
        group_nums = [2, 2, 2, 2]
        group_dims = [64, 128, 256, 512]
        self.resblocks = nn.ModuleList([])

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=group_dims[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(group_dims[0]),
            nn.ReLU()
        )

        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)

        for group_i, group_num in enumerate(group_nums):
            for block_i in range(group_num):
                if block_i != 0:
                    self.resblocks.append(PreactivateBlock(group_dims[group_i], group_dims[group_i], downsample=False, firstact=True))
                elif group_i == 0 and block_i == 0:
                    self.resblocks.append(PreactivateBlock(group_dims[group_i], group_dims[group_i], downsample=False, firstact=False))
                elif group_i != 0 and block_i == 0:
                    self.resblocks.append(PreactivateBlock(group_dims[group_i-1], group_dims[group_i], downsample=True, firstact=True))
                else:
                    raise Exception("Wrong building resblocks")
        
        self.fc1 = nn.Sequential(
            nn.BatchNorm2d(group_dims[-1]),
            nn.ReLU(),
            nn.Conv2d(group_dims[-1], out_channels=self.emb_dim, kernel_size=(16, 1), stride=1, padding=0),
            nn.BatchNorm2d(self.emb_dim)
        )

        self.pooltime = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    
    def forward(self, x, y):
        out = x.permute(0,2,1)
        out = out.unsqueeze(1)
        out = self.conv1(out)
        out = self.pool1(out)

        for layer in self.resblocks:
            out = layer(out)
            # print(out.shape)
        
        out = self.fc1(out)
        # print(out.shape)
        out = self.pooltime(out)
        # print(out.shape)
        out = out.squeeze(-1)
        out = out.squeeze(-1)
        return out

class Resnet18_Maxpool(nn.Module):
    def __init__(self, feat_dim, emb_dim):
        super(Resnet18_Maxpool, self).__init__()
        self.emb_dim = emb_dim
        group_nums = [2, 2, 2, 2]
        group_dims = [64, 128, 256, 512]
        self.resblocks = nn.ModuleList([])

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=group_dims[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(group_dims[0]),
            nn.ReLU()
        )

        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)

        for group_i, group_num in enumerate(group_nums):
            for block_i in range(group_num):
                if block_i != 0:
                    self.resblocks.append(PreactivateMaxpoolBlock(group_dims[group_i], group_dims[group_i], downsample=False, firstact=True))
                elif group_i == 0 and block_i == 0:
                    self.resblocks.append(PreactivateMaxpoolBlock(group_dims[group_i], group_dims[group_i], downsample=False, firstact=False))
                elif group_i != 0 and block_i == 0:
                    self.resblocks.append(PreactivateMaxpoolBlock(group_dims[group_i-1], group_dims[group_i], downsample=True, firstact=True))
                else:
                    raise Exception("Wrong building resblocks")
        
        self.fc1 = nn.Sequential(
            nn.BatchNorm2d(group_dims[-1]),
            nn.ReLU(),
            nn.Conv2d(group_dims[-1], out_channels=self.emb_dim, kernel_size=(16, 1), stride=1, padding=0),
            nn.BatchNorm2d(self.emb_dim)
        )

        self.pooltime = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    
    def forward(self, x, y):
        out = x.permute(0,2,1)
        out = out.unsqueeze(1)
        out = self.conv1(out)
        out = self.pool1(out)

        for layer in self.resblocks:
            out = layer(out)
            # print(out.shape)
        
        out = self.fc1(out)
        # print(out.shape)
        out = self.pooltime(out)
        # print(out.shape)
        out = out.squeeze(-1)
        out = out.squeeze(-1)
        return out

class SeResnet18(nn.Module):
    def __init__(self, feat_dim, emb_dim):
        super(SeResnet18, self).__init__()
        group_nums = [2, 2, 2, 2]
        group_dims = [64, 128, 256, 512]
        self.resblocks = nn.ModuleList([])

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=group_dims[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(group_dims[0]),
            nn.ReLU()
        )

        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)

        for group_i, group_num in enumerate(group_nums):
            for block_i in range(group_num):
                if block_i != 0:
                    self.resblocks.append(SePreactivateBlock(group_dims[group_i], group_dims[group_i], downsample=False, firstact=True))
                elif group_i == 0 and block_i == 0:
                    self.resblocks.append(SePreactivateBlock(group_dims[group_i], group_dims[group_i], downsample=False, firstact=False))
                elif group_i != 0 and block_i == 0:
                    self.resblocks.append(SePreactivateBlock(group_dims[group_i-1], group_dims[group_i], downsample=True, firstact=True))
                else:
                    raise Exception("Wrong building resblocks")
        
        self.fc1 = nn.Sequential(
            nn.BatchNorm2d(group_dims[-1]),
            nn.ReLU(),
            nn.Conv2d(group_dims[-1], out_channels=512, kernel_size=(16, 1), stride=1, padding=0),
            nn.BatchNorm2d(512)
        )

        self.pooltime = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    
    def forward(self, x, y):
        out = x.permute(0,2,1)
        out = out.unsqueeze(1)
        out = self.conv1(out)
        out = self.pool1(out)

        for layer in self.resblocks:
            out = layer(out)
            # print(out.shape)
        
        out = self.fc1(out)
        # print(out.shape)
        out = self.pooltime(out)
        # print(out.shape)
        out = out.squeeze(-1)
        out = out.squeeze(-1)
        return out


class IBNResnet18(nn.Module):
    def __init__(self, feat_dim, emb_dim):
        super(IBNResnet18, self).__init__()
        group_nums = [2, 2, 2, 2]
        group_dims = [64, 128, 256, 512]
        self.resblocks = nn.ModuleList([])

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=group_dims[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(group_dims[0]),
            nn.ReLU()
        )

        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)

        for group_i, group_num in enumerate(group_nums):
            for block_i in range(group_num):
                if block_i != 0:
                    if group_i < 2:
                        self.resblocks.append(IBNPreactivateBlock(group_dims[group_i], group_dims[group_i], downsample=False, firstact=True))
                    else:
                        self.resblocks.append(PreactivateBlock(group_dims[group_i], group_dims[group_i], downsample=False, firstact=True))
                
                elif group_i == 0 and block_i == 0:
                    self.resblocks.append(IBNPreactivateBlock(group_dims[group_i], group_dims[group_i], downsample=False, firstact=False))
                
                elif group_i != 0 and block_i == 0:
                    if group_i < 2:
                        self.resblocks.append(IBNPreactivateBlock(group_dims[group_i-1], group_dims[group_i], downsample=True, firstact=True))
                    else:
                        self.resblocks.append(PreactivateBlock(group_dims[group_i-1], group_dims[group_i], downsample=True, firstact=True))
                else:
                    raise Exception("Wrong building resblocks")
        
        self.fc1 = nn.Sequential(
            nn.BatchNorm2d(group_dims[-1]),
            nn.ReLU(),
            nn.Conv2d(group_dims[-1], out_channels=512, kernel_size=(16, 1), stride=1, padding=0),
            nn.BatchNorm2d(512)
        )

        self.pooltime = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    
    def forward(self, x, y):
        out = x.permute(0,2,1)
        out = out.unsqueeze(1)
        out = self.conv1(out)
        out = self.pool1(out)

        for layer in self.resblocks:
            out = layer(out)
            # print(out.shape)
        
        out = self.fc1(out)
        # print(out.shape)
        out = self.pooltime(out)
        # print(out.shape)
        out = out.squeeze(-1)
        out = out.squeeze(-1)
        return out

class Linear_softmax_ce_head(nn.Module):
    def __init__(self, backbone, model_settings):
        super(Linear_softmax_ce_head, self).__init__()
        self.backbone = nn.DataParallel(backbone)
        self.linear = nn.Linear(model_settings['emb_size'], model_settings['class_num'])
        self.loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, x, y, mod):
        emb = self.backbone(x, y)
        logits = self.linear(emb)
        loss = self.loss(logits, y)

        pred = logits.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        label = y.data.cpu().numpy()        
        acc = np.mean((pred == label).astype(int))

        return loss, logits, emb, acc, 0.0      

# class Linear_softmax_ce_head_nobias(nn.Module):
#     def __init__(self, backbone, model_settings):
#         super(Linear_softmax_ce_head_nobias, self).__init__()
#         self.backbone = nn.DataParallel(backbone)
#         self.linear = nn.Linear(model_settings['emb_size'], model_settings['class_num'], bias=False)
#         self.loss = torch.nn.CrossEntropyLoss()
    
#     def forward(self, x, y):
#         emb = self.backbone(x, y)
#         logits = self.linear(emb)
#         loss = self.loss(logits, y)

#         return loss, logits, emb  

class AM_softmax_anneal_ce_head(nn.Module):
    def __init__(self, backbone, model_settings):
        super(AM_softmax_anneal_ce_head, self).__init__()
        self.th_step = 28000.0
        self.iter = 0.0
        self.max_m = model_settings['m']
        self.model_settings = model_settings

        self.backbone = nn.DataParallel(backbone)
        self.metrics = AMSoftmax(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], s=model_settings['s'], m=model_settings['m'])
        self.loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, x, y):
        if y.size()[0] != 1:
            self.iter += 1.0
            m = min(self.max_m, (self.iter / self.th_step) * self.model_settings['m'])
        else:
            m = 0.0

        emb = self.backbone(x, y)
        logits = self.metrics(emb, y, s=self.model_settings['s'], m=m)
        loss = self.loss(logits, y)

        pred = logits.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        label = y.data.cpu().numpy()        
        acc = np.mean((pred == label).astype(int))

        return loss, logits, emb, acc, 0.0

class AM_normfree_softmax_anneal_ce_head(nn.Module):
    def __init__(self, backbone, model_settings):
        super(AM_normfree_softmax_anneal_ce_head, self).__init__()
        self.th_step = 28000.0
        self.iter = 0.0
        self.max_m = model_settings['m']
        self.model_settings = model_settings
        
        self.backbone = nn.DataParallel(backbone)
        self.metrics = AMSoftmax_normfree(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], s=model_settings['s'], m=model_settings['m'])
        self.loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, x, y):
        if y.size()[0] != 1:
            self.iter += 1.0
            m = min(self.max_m, (self.iter / self.th_step) * self.model_settings['m'])
        else:
            m = 0.0

        emb = self.backbone(x, y)
        logits, nm_W = self.metrics(emb, y, s=self.model_settings['s'], m=m)
        loss = self.loss(logits, y)

        pred = logits.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        label = y.data.cpu().numpy()        
        acc = np.mean((pred == label).astype(int))

        return loss, logits, emb, acc, 0.0

# finetune
# class AM_normfree_softmax_anneal_inter_ce_head(nn.Module):
#     def __init__(self, backbone, model_settings):
#         super(AM_normfree_softmax_anneal_inter_ce_head, self).__init__()
#         # self.th_step = 28000.0
#         # self.iter = 0.0
#         self.max_m = model_settings['m']
#         self.model_settings = model_settings
        
#         self.backbone = nn.DataParallel(backbone)
#         self.metrics = AMSoftmax_normfree(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], s=model_settings['s'], m=model_settings['m'])
#         self.loss = torch.nn.CrossEntropyLoss()
    
#         self.I = torch.eye(self.model_settings['class_num']).cuda()
    
#     def forward(self, x, y):
#         if y.size()[0] != 1:
#             m = self.max_m
#         else:
#             m = 0.0

#         emb = self.backbone(x, y)
#         logits, nm_W = self.metrics(emb, y, s=self.model_settings['s'], m=m)
#         loss = self.loss(logits, y)

#         if y.size(0) != 1:
#             A = torch.mm(nm_W, nm_W.t())
#             A = A.clamp(min=0)
#             B = torch.mm((A-self.I), (A-self.I).t())
#             inter_loss = torch.trace(B)/self.model_settings['class_num']
#             loss = (1-self.model_settings['lmd_inter'])*loss + self.model_settings['lmd_inter']*inter_loss
#             inter_loss_out = inter_loss.item()
#         else:
#             inter_loss_out = None

#         pred = logits.data.cpu().numpy()
#         pred = np.argmax(pred, axis=1)
#         label = y.data.cpu().numpy()        
#         acc = np.mean((pred == label).astype(int))

#         return loss, logits, emb, acc, inter_loss_out

# # normal
class AM_normfree_softmax_anneal_inter_ce_head(nn.Module):
    def __init__(self, backbone, model_settings):
        super(AM_normfree_softmax_anneal_inter_ce_head, self).__init__()
        self.th_step = 28000.0
        self.iter = 0.0
        self.max_m = model_settings['m']
        self.model_settings = model_settings
        
        self.backbone = nn.DataParallel(backbone)
        self.metrics = AMSoftmax_normfree(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], s=model_settings['s'], m=model_settings['m'])
        self.loss = torch.nn.CrossEntropyLoss()
    
        self.I = torch.eye(self.model_settings['class_num']).cuda()
    
    def forward(self, x, y):
        if y.size()[0] != 1:
            self.iter += 1.0
            m = min(self.max_m, (self.iter / self.th_step) * self.model_settings['m'])
        else:
            m = 0.0

        emb = self.backbone(x, y)
        logits, nm_W = self.metrics(emb, y, s=self.model_settings['s'], m=m)
        loss = self.loss(logits, y)

        if y.size(0) != 1:
            A = torch.mm(nm_W, nm_W.t())
            A = A.clamp(min=0)
            B = torch.mm((A-self.I), (A-self.I).t())
            inter_loss = torch.trace(B)/self.model_settings['class_num']
            loss = (1-self.model_settings['lmd_inter'])*loss + self.model_settings['lmd_inter']*inter_loss
            inter_loss_out = inter_loss.item()
        else:
            inter_loss_out = None

        pred = logits.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        label = y.data.cpu().numpy()        
        acc = np.mean((pred == label).astype(int))

        return loss, logits, emb, acc, inter_loss_out

class AAM_softmax_anneal_ce_head(nn.Module):
    def __init__(self, backbone, model_settings):
        super(AAM_softmax_anneal_ce_head, self).__init__()
        self.th_step = 28000.0
        self.iter = 0.0
        self.max_m = model_settings['m']
        self.model_settings = model_settings

        self.backbone = nn.DataParallel(backbone)
        self.metrics = AAMSoftmax(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], s=model_settings['s'], m=model_settings['m'])
        self.loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, x, y):
        if y.size()[0] != 1:
            self.iter += 1.0
            m = min(self.max_m, (self.iter / self.th_step) * self.model_settings['m'])
        else:
            m = 0.0
        emb = self.backbone(x, y)
        logits = self.metrics(emb, y, s=self.model_settings['s'], m=m)
        loss = self.loss(logits, y)

        pred = logits.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        label = y.data.cpu().numpy()        
        acc = np.mean((pred == label).astype(int))

        return loss, logits, emb, acc, 0.0

# class AAM_m_softmax_ce_head(nn.Module):
#     def __init__(self, backbone, model_settings):
#         super(AAM_m_softmax_ce_head, self).__init__()
#         self.backbone = nn.DataParallel(backbone)
#         self.metrics = AAMSoftmax_m(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], s=model_settings['s'], m=model_settings['m'])
#         self.loss = torch.nn.CrossEntropyLoss()
    
#     def forward(self, x, y, s, m):
#         emb = self.backbone(x, y)
#         logits = self.metrics(emb, y, s=s, m=m)
#         loss = self.loss(logits, y)

#         pred = logits.data.cpu().numpy()
#         pred = np.argmax(pred, axis=1)
#         label = y.data.cpu().numpy()        
#         acc = np.mean((pred == label).astype(int))

#         return loss, logits, emb

# class AAM_normfree_softmax_ce_head(nn.Module):
#     def __init__(self, backbone, model_settings):
#         super(AAM_normfree_softmax_ce_head, self).__init__()
#         self.backbone = nn.DataParallel(backbone)
#         self.metrics = AAMSoftmax_normfree(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], s=model_settings['s'], m=model_settings['m'])
#         self.loss = torch.nn.CrossEntropyLoss()
    
#     def forward(self, x, y, s, m):
#         emb = self.backbone(x, y)
#         logits = self.metrics(emb, y, s=s, m=m)
#         loss = self.loss(logits, y)

#         return loss, logits, emb 

class AAM_normfree_softmax_anneal_ce_head(nn.Module):
    def __init__(self, backbone, model_settings):
        super(AAM_normfree_softmax_anneal_ce_head, self).__init__()
        self.th_step = 28000.0
        self.iter = 0.0
        self.max_m = model_settings['m']
        self.model_settings = model_settings

        self.backbone = nn.DataParallel(backbone)
        self.metrics = AAMSoftmax_normfree(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], s=model_settings['s'], m=model_settings['m'])
        self.loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, x, y):
        if y.size()[0] != 1:
            self.iter += 1.0
            m = min(self.max_m, (self.iter / self.th_step) * self.model_settings['m'])
        else:
            m = 0.0

        emb = self.backbone(x, y)
        logits, nm_W = self.metrics(emb, y, s=self.model_settings['s'], m=m)
        loss = self.loss(logits, y)

        pred = logits.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        label = y.data.cpu().numpy()        
        acc = np.mean((pred == label).astype(int))

        return loss, logits, emb, acc, 0

# Finetune
# class AAM_normfree_softmax_anneal_inter_ce_head(nn.Module):
#     def __init__(self, backbone, model_settings):
#         super(AAM_normfree_softmax_anneal_inter_ce_head, self).__init__()
#         self.th_step = 28000.0
#         self.iter = 0.0
#         self.max_m = model_settings['m']
#         self.model_settings = model_settings

#         self.backbone = nn.DataParallel(backbone)
#         self.metrics = AAMSoftmax_normfree(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], s=model_settings['s'], m=model_settings['m'])
#         self.loss = torch.nn.CrossEntropyLoss()

#         self.I = torch.eye(self.model_settings['class_num']).cuda()
    
#     def forward(self, x, y):
#         if y.size(0) != 1:
#             m = self.max_m
#         else:
#             m = 0.0

#         emb = self.backbone(x, y)
#         logits, nm_W = self.metrics(emb, y, s=self.model_settings['s'], m=m)
#         loss = self.loss(logits, y)

#         if y.size(0) != 1:
#             A = torch.mm(nm_W, nm_W.t())
#             A = A.clamp(min=0)
#             B = torch.mm((A-self.I), (A-self.I).t())
#             inter_loss = torch.trace(B)/self.model_settings['class_num']
#             loss = (1-self.model_settings['lmd_inter'])*loss + self.model_settings['lmd_inter']*inter_loss
#             inter_loss_out = inter_loss.item()
#         else:
#             inter_loss_out = None

#         pred = logits.data.cpu().numpy()
#         pred = np.argmax(pred, axis=1)
#         label = y.data.cpu().numpy()        
#         acc = np.mean((pred == label).astype(int))

#         return loss, logits, emb, acc, inter_loss_out

# Normal
class AAM_normfree_softmax_anneal_inter_ce_head(nn.Module):
    def __init__(self, backbone, model_settings):
        super(AAM_normfree_softmax_anneal_inter_ce_head, self).__init__()
        self.th_step = 28000.0
        self.iter = 0.0
        self.max_m = model_settings['m']
        self.model_settings = model_settings

        self.backbone = nn.DataParallel(backbone)
        self.metrics = AAMSoftmax_normfree(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], s=model_settings['s'], m=model_settings['m'])
        self.loss = torch.nn.CrossEntropyLoss()

        self.I = torch.eye(self.model_settings['class_num']).cuda()
    
    def forward(self, x, y):
        if y.size(0) != 1:
            self.iter += 1.0
            m = min(self.max_m, (self.iter / self.th_step) * self.model_settings['m'])
        else:
            m = 0.0

        emb = self.backbone(x, y)
        logits, nm_W = self.metrics(emb, y, s=self.model_settings['s'], m=m)
        loss = self.loss(logits, y)

        if y.size(0) != 1:
            A = torch.mm(nm_W, nm_W.t())
            A = A.clamp(min=0)
            B = torch.mm((A-self.I), (A-self.I).t())
            inter_loss = torch.trace(B)/self.model_settings['class_num']
            loss = (1-self.model_settings['lmd_inter'])*loss + self.model_settings['lmd_inter']*inter_loss
            inter_loss_out = inter_loss.item()
        else:
            inter_loss_out = None

        pred = logits.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        label = y.data.cpu().numpy()        
        acc = np.mean((pred == label).astype(int))

        return loss, logits, emb, acc, inter_loss_out


class A_softmax_ce_head(nn.Module):
    def __init__(self, backbone, model_settings):
        super(A_softmax_ce_head, self).__init__()
        self.backbone = nn.DataParallel(backbone)
        self.metrics = ASoftmax(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], m=model_settings['m'], phiflag=True)
        self.loss = AngleLoss()
    
    def forward(self, x, y):
        emb = self.backbone(x, y)
        logits, nm_W = self.metrics(emb)
        loss = self.loss(logits, y)

        pred = logits[0].data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        label = y.data.cpu().numpy()        
        acc = np.mean((pred == label).astype(int))

        return loss, logits[0], emb, acc, 0

# class A_softmax_inter_ce_head(nn.Module):
#     def __init__(self, backbone, model_settings):
#         super(A_softmax_inter_ce_head, self).__init__()
#         self.backbone = nn.DataParallel(backbone)
#         self.metrics = ASoftmax(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], m=model_settings['m'], phiflag=True)
#         self.loss = AngleLoss()
#         self.model_settings = model_settings
#         self.I = torch.eye(self.model_settings['class_num']).cuda()
    
#     def forward(self, x, y):
#         emb = self.backbone(x, y)
#         logits, W = self.metrics(emb)
#         loss = self.loss(logits, y)
#         # W = self.metrics.weight
#         # W = W.renorm(2,1,1e-5).mul(1e5)
#         A = torch.mm(W.t(), W)
#         A = A.clamp(min=0)
#         # I = torch.eye(self.model_settings['class_num']).cuda()
#         B = torch.mm((A-self.I), (A-self.I).t())
#         inter_loss = torch.trace(B)/self.model_settings['class_num']
#         loss = (1-self.model_settings['lmd_inter'])*loss + self.model_settings['lmd_inter']*inter_loss

#         pred = logits[0].data.cpu().numpy()
#         pred = np.argmax(pred, axis=1)
#         label = y.data.cpu().numpy()        
#         acc = np.mean((pred == label).astype(int))

#         return loss, logits[0], emb, acc, inter_loss.item()

class A_softmax_inter_ce_head(nn.Module):
    def __init__(self, backbone, model_settings):
        super(A_softmax_inter_ce_head, self).__init__()
        self.backbone = nn.DataParallel(backbone)
        self.metrics = ASoftmax(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], m=model_settings['m'], phiflag=True)
        self.loss = AngleLoss()
        self.model_settings = model_settings
        self.I = torch.eye(self.model_settings['class_num']).cuda()
    
    def forward(self, x, y):
        emb = self.backbone(x, y)
        logits, nm_W = self.metrics(emb)
        loss = self.loss(logits, y)
        if y.size(0) != 1:
            A = torch.mm(nm_W.t(), nm_W)
            A = A.clamp(min=0)
            B = torch.mm((A-self.I), (A-self.I).t())
            inter_loss = torch.trace(B)/self.model_settings['class_num']
            loss = (1-self.model_settings['lmd_inter'])*loss + self.model_settings['lmd_inter']*inter_loss
            inter_loss_out = inter_loss.item()
        else:
            inter_loss_out = None

        pred = logits[0].data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        label = y.data.cpu().numpy()        
        acc = np.mean((pred == label).astype(int))

        return loss, logits[0], emb, acc, inter_loss_out

class A_softmax_mixup_ce_head(nn.Module):
    def __init__(self, backbone, model_settings):
        super(A_softmax_mixup_ce_head, self).__init__()
        self.backbone = nn.DataParallel(backbone)
        self.metrics = ASoftmax(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], m=model_settings['m'], phiflag=True)
        self.loss = AngleLoss()
        self.model_settings = model_settings
    
    def forward(self, x, y):
        # inputs, targets_a, targets_b, lam = mixup_data(x, y, args.alpha, use_cuda)
        # make mixup
        lam = np.random.beta(self.model_settings['alpha'], self.model_settings['alpha'])
        batch_size = x.size()[0]
        index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index,:]
        target_a, target_b = y, y[index]

        emb = self.backbone(mixed_x, target_a) # the backbone dont need label, randomly give it one
        logits = self.metrics(emb)
        loss_a = self.loss(logits, target_a)
        loss_b = self.loss(logits, target_b)
        loss = lam * loss_a + (1 - lam) * loss_b

        pred = logits[0].data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        label_a = target_a.data.cpu().numpy()
        label_b = target_b.data.cpu().numpy()        
        acc_a = np.mean((pred == label_a).astype(int))
        acc_b = np.mean((pred == label_b).astype(int))
        acc = lam * acc_a + (1 - lam) * acc_b

        return loss, logits[0], emb, acc

class A_softmax_mixup_inter_ce_head(nn.Module):
    def __init__(self, backbone, model_settings):
        super(A_softmax_mixup_inter_ce_head, self).__init__()
        self.backbone = nn.DataParallel(backbone)
        self.metrics = ASoftmax(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], m=model_settings['m'], phiflag=True)
        self.loss = AngleLoss()
        self.model_settings = model_settings
        self.I = torch.eye(self.model_settings['class_num']).cuda()
    
    def forward(self, x, y):
        # inputs, targets_a, targets_b, lam = mixup_data(x, y, args.alpha, use_cuda)
        # make mixup
        lam = np.random.beta(self.model_settings['alpha'], self.model_settings['alpha'])
        batch_size = x.size()[0]
        index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index,:]
        target_a, target_b = y, y[index]

        emb = self.backbone(mixed_x, target_a) # the backbone dont need label, randomly give it one
        logits, nm_W = self.metrics(emb)
        loss_a = self.loss(logits, target_a)
        loss_b = self.loss(logits, target_b)
        loss = lam * loss_a + (1 - lam) * loss_b

        if y.size(0) != 1:
            A = torch.mm(nm_W.t(), nm_W)
            A = A.clamp(min=0)
            B = torch.mm((A-self.I), (A-self.I).t())
            inter_loss = torch.trace(B)/self.model_settings['class_num']
            loss = (1-self.model_settings['lmd_inter'])*loss + self.model_settings['lmd_inter']*inter_loss
            inter_loss_out = inter_loss.item()
        else:
            inter_loss_out = None

        pred = logits[0].data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        label_a = target_a.data.cpu().numpy()
        label_b = target_b.data.cpu().numpy()        
        acc_a = np.mean((pred == label_a).astype(int))
        acc_b = np.mean((pred == label_b).astype(int))
        acc = lam * acc_a + (1 - lam) * acc_b

        return loss, logits[0], emb, acc, inter_loss_out


class A_softmax_inter_a_ce_head(nn.Module):
    def __init__(self, backbone, model_settings):
        super(A_softmax_inter_a_ce_head, self).__init__()
        self.backbone = nn.DataParallel(backbone)
        self.metrics = ASoftmax(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], m=model_settings['m'], phiflag=True)
        self.loss = AngleLoss()
        self.model_settings = model_settings
        self.I = torch.eye(self.model_settings['class_num']).cuda()
    
    def forward(self, x, y):
        emb = self.backbone(x, y)
        logits, nm_W = self.metrics(emb)
        loss = self.loss(logits, y)
        if y.size(0) != 1:
            A = torch.mm(nm_W.t(), nm_W)
            # A = A.clamp(min=0)
            B = torch.mm(((A+1)/2.0-self.I), ((A+1)/2.0-self.I).t())
            # B = torch.mm((A-self.I), (A-self.I).t())
            inter_loss = torch.trace(B)/(self.model_settings['class_num'] * self.model_settings['class_num'] - self.model_settings['class_num'])
            loss = (1-self.model_settings['lmd_inter'])*loss + self.model_settings['lmd_inter']*inter_loss
            inter_loss_out = inter_loss.item()
        else:
            inter_loss_out = None

        pred = logits[0].data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        label = y.data.cpu().numpy()        
        acc = np.mean((pred == label).astype(int))

        return loss, logits[0], emb, acc, inter_loss_out
    
# class A_softmax_MHE_ce_head(nn.Module):
#     def __init__(self, backbone, model_settings):
#         super(A_softmax_MHE_ce_head, self).__init__()
#         self.backbone = nn.DataParallel(backbone)
#         self.metrics = ASoftmax(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], m=model_settings['m'], phiflag=True)
#         self.loss = AngleLoss()
#         self.model_settings = model_settings
#         self.I = torch.eye(self.model_settings['class_num']).cuda()
    
#     def forward(self, x, y):
#         emb = self.backbone(x, y)
#         logits, nm_W = self.metrics(emb)
#         loss = self.loss(logits, y)
        
#         if y.size(0) != 1:
#             w_selector = torch.zeros(self.model_settings['class_num'], y.size(0)).cuda()
#             w_selector.scatter_(0, y.view(1, -1), 1)
#             ws = nm_W.mm(w_selector)
#             nm_W_repeats = nm_W.repeat(1, y.size(0))
#             ws_repeats = ws.view(-1, 1).repeat(1, self.model_settings['class_num']).view(self.model_settings['emb_size'], y.size(0)*self.model_settings['class_num'])
#             z = ws_repeats - nm_W_repeats
#             z_n = torch.norm(z, dim=0)
#             z_n_1 = 1/(z_n[z_n.nonzero()])**2
#             inter_loss = torch.sum(z_n_1) / (y.size(0)*(self.model_settings['class_num']-1))
#             loss = (1-self.model_settings['lmd_inter'])*loss + self.model_settings['lmd_inter']*inter_loss
#             inter_loss_out = inter_loss.item()
#         else:
#             inter_loss_out = None

#         pred = logits[0].data.cpu().numpy()
#         pred = np.argmax(pred, axis=1)
#         label = y.data.cpu().numpy()        
#         acc = np.mean((pred == label).astype(int))

#         return loss, logits[0], emb, acc, inter_loss_out, nm_W, y

class A_softmax_MHE_ce_head(nn.Module):
    def __init__(self, backbone, model_settings):
        super(A_softmax_MHE_ce_head, self).__init__()
        self.backbone = nn.DataParallel(backbone)
        self.metrics = ASoftmax(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], m=model_settings['m'], phiflag=True)
        self.loss = AngleLoss()
        self.model_settings = model_settings
        self.I = torch.eye(self.model_settings['class_num']).cuda()
    
    def forward(self, x, y):
        emb = self.backbone(x, y)
        logits, nm_W = self.metrics(emb)
        loss = self.loss(logits, y)
        
        if y.size(0) != 1:
            w_selector = torch.zeros(self.model_settings['class_num'], y.size(0)).cuda()
            w_selector.scatter_(0, y.view(1, -1), 1)
            ws = nm_W.mm(w_selector)
            nm_W_repeats = nm_W.repeat(1, y.size(0))
            ws_repeats = ws.view(-1, 1).repeat(1, self.model_settings['class_num']).view(self.model_settings['emb_size'], y.size(0)*self.model_settings['class_num'])
            z = ws_repeats - nm_W_repeats
            z_n = torch.norm(z, dim=0)
            z_n_1 = 1/(z_n[z_n.nonzero()])**2 # normal
            # z_n_1 = 1/(z_n[z_n.nonzero()]) # a
            # z_n_1 = -torch.log(z_n[z_n.nonzero()]) # a2
            inter_loss = torch.sum(z_n_1) / (y.size(0)*(self.model_settings['class_num']-1))
            loss = loss + self.model_settings['lmd_inter']*inter_loss
            inter_loss_out = inter_loss.item()
        else:
            inter_loss_out = None

        pred = logits[0].data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        label = y.data.cpu().numpy()        
        acc = np.mean((pred == label).astype(int))

        return loss, logits[0], emb, acc, inter_loss_out

class AM_normfree_softmax_anneal_MHE_ce_head(nn.Module):
    def __init__(self, backbone, model_settings, opt):
        super(AM_normfree_softmax_anneal_MHE_ce_head, self).__init__()
        self.th_step = opt.th_epochs * opt.steps_per_epoch
        self.iter = 0.0
        self.max_m = model_settings['m']
        self.model_settings = model_settings
        
        self.backbone = nn.DataParallel(backbone)
        self.metrics = AMSoftmax_normfree(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], s=model_settings['s'], m=model_settings['m'])
        self.loss = torch.nn.CrossEntropyLoss()
    
        self.I = torch.eye(self.model_settings['class_num']).cuda()
    
    def forward(self, x, y):
        if y.size()[0] != 1:
            self.iter += 1.0
            m = min(self.max_m, (self.iter / self.th_step) * self.model_settings['m'])
        else:
            m = 0.0

        emb = self.backbone(x, y)
        logits, nm_W = self.metrics(emb, y, s=self.model_settings['s'], m=m)
        loss = self.loss(logits, y)

        if y.size(0) != 1:
            nm_W = nm_W.t()
            w_selector = torch.zeros(self.model_settings['class_num'], y.size(0)).cuda()
            w_selector.scatter_(0, y.view(1, -1), 1)
            ws = nm_W.mm(w_selector)
            nm_W_repeats = nm_W.repeat(1, y.size(0))
            ws_repeats = ws.view(-1, 1).repeat(1, self.model_settings['class_num']).view(self.model_settings['emb_size'], y.size(0)*self.model_settings['class_num'])
            z = ws_repeats - nm_W_repeats
            z_n = torch.norm(z, dim=0)
            z_n_1 = 1/(z_n[z_n.nonzero()])**2 # normal
            # z_n_1 = 1/(z_n[z_n.nonzero()]) # a
            # z_n_1 = -torch.log(z_n[z_n.nonzero()]) # a2
            inter_loss = torch.sum(z_n_1) / (y.size(0)*(self.model_settings['class_num']-1))
            loss = loss + self.model_settings['lmd_inter']*inter_loss
            inter_loss_out = inter_loss.item()
        else:
            inter_loss_out = None

        pred = logits.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        label = y.data.cpu().numpy()        
        acc = np.mean((pred == label).astype(int))

        return loss, logits, emb, acc, inter_loss_out