import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')

import torch
import torch.nn as nn
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
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=1, stride=2, padding=0)
        )

    def forward(self, x):
        residual = x
        if self.firstact:
            out = self.bn1(x)
            out = self.relu1(out)
            out = self.conv1(out)
        else:
            out = self.conv1(x)

        out = self.bn2(out)
        out = self.relu2(out)        
        out = self.conv2(out)

        if self.downsample is True:
            residual = self.downsampler(x)

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
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=1, stride=2, padding=0)
        )

    def forward(self, x):
        residual = x
        if self.firstact:
            out = self.bn1(x)
            out = self.relu1(out)
            out = self.conv1(out)
        else:
            out = self.conv1(x)

        out = self.bn2(out)
        out = self.relu2(out)        
        out = self.conv2(out)

        if self.downsample is True:
            residual = self.downsampler(x)

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
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
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
            out = self.conv1(out)
        else:
            out = self.conv1(x)

        out = self.bn2(out)
        out = self.relu2(out)        
        out = self.conv2(out)

        if self.downsample is True:
            residual = self.downsampler(x)

        original_out = out
        out = self.globalAvgPool(out)
        out = out.view(out.shape[0], -1)
        out = self.SE_Block(out)
        out = out.view(out.shape[0], out.shape[1], 1, 1)
        out = original_out * out

        out += residual
        return out


# class Resnet18(nn.Module):
#     def __init__(self, feat_dim, emb_dim):
#         super(Resnet18, self).__init__()
#         group_nums = [2, 2, 2, 2]
#         group_dims = [64, 128, 256, 512]
#         self.resblocks = nn.ModuleList([])

#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=group_dims[0], kernel_size=7, stride=2, padding=3),
#             nn.BatchNorm2d(group_dims[0]),
#             nn.ReLU()
#         )

#         self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)

#         for group_i, group_num in enumerate(group_nums):
#             for block_i in range(group_num):
#                 if block_i != 0:
#                     self.resblocks.append(PreactivateBlock(group_dims[group_i], group_dims[group_i], downsample=False, firstact=True))
#                 elif group_i == 0 and block_i == 0:
#                     self.resblocks.append(PreactivateBlock(group_dims[group_i], group_dims[group_i], downsample=False, firstact=False))
#                 elif group_i != 0 and block_i == 0:
#                     self.resblocks.append(PreactivateBlock(group_dims[group_i-1], group_dims[group_i], downsample=True, firstact=True))
#                 else:
#                     raise Exception("Wrong building resblocks")
        
#         self.fc1 = nn.Sequential(
#             nn.BatchNorm2d(group_dims[-1]),
#             nn.ReLU(),
#             nn.Conv2d(group_dims[-1], out_channels=512, kernel_size=(16, 1), stride=1, padding=0),
#             nn.BatchNorm2d(512)
#         )

#         self.pooltime = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    
#     def forward(self, x, y):
#         out = x.permute(0,2,1)
#         out = out.unsqueeze(1)
#         out = self.conv1(out)
#         out = self.pool1(out)

#         for layer in self.resblocks:
#             out = layer(out)
#             # print(out.shape)
        
#         out = self.fc1(out)
#         # print(out.shape)
#         out = self.pooltime(out)
#         # print(out.shape)
#         out = out.squeeze(-1)
#         out = out.squeeze(-1)
#         return out

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
    
    def forward(self, x, y):
        emb = self.backbone(x, y)
        logits = self.linear(emb)
        loss = self.loss(logits, y)

        return loss, logits, emb       

class Linear_softmax_ce_head_nobias(nn.Module):
    def __init__(self, backbone, model_settings):
        super(Linear_softmax_ce_head_nobias, self).__init__()
        self.backbone = nn.DataParallel(backbone)
        self.linear = nn.Linear(model_settings['emb_size'], model_settings['class_num'], bias=False)
        self.loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, x, y):
        emb = self.backbone(x, y)
        logits = self.linear(emb)
        loss = self.loss(logits, y)

        return loss, logits, emb  

class AM_softmax_ce_head(nn.Module):
    def __init__(self, backbone, model_settings):
        super(AM_softmax_ce_head, self).__init__()
        self.backbone = nn.DataParallel(backbone)
        self.metrics = AMSoftmax(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], s=model_settings['s'], m=model_settings['m'])
        self.loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, x, y, s, m):
        emb = self.backbone(x, y)
        logits = self.metrics(emb, y, s=s, m=m)
        loss = self.loss(logits, y)

        return loss, logits, emb

class AAM_softmax_ce_head(nn.Module):
    def __init__(self, backbone, model_settings):
        super(AAM_softmax_ce_head, self).__init__()
        self.backbone = nn.DataParallel(backbone)
        self.metrics = AAMSoftmax(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], s=model_settings['s'], m=model_settings['m'])
        self.loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, x, y, s, m):
        emb = self.backbone(x, y)
        logits = self.metrics(emb, y, s=s, m=m)
        loss = self.loss(logits, y)

        return loss, logits, emb  

class AAM_m_softmax_ce_head(nn.Module):
    def __init__(self, backbone, model_settings):
        super(AAM_m_softmax_ce_head, self).__init__()
        self.backbone = nn.DataParallel(backbone)
        self.metrics = AAMSoftmax_m(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], s=model_settings['s'], m=model_settings['m'])
        self.loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, x, y, s, m):
        emb = self.backbone(x, y)
        logits = self.metrics(emb, y, s=s, m=m)
        loss = self.loss(logits, y)

        return loss, logits, emb 

class AAM_normfree_softmax_ce_head(nn.Module):
    def __init__(self, backbone, model_settings):
        super(AAM_normfree_softmax_ce_head, self).__init__()
        self.backbone = nn.DataParallel(backbone)
        self.metrics = AAMSoftmax_normfree(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], s=model_settings['s'], m=model_settings['m'])
        self.loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, x, y, s, m):
        emb = self.backbone(x, y)
        logits = self.metrics(emb, y, s=s, m=m)
        loss = self.loss(logits, y)

        return loss, logits, emb 

class AAM_normfree_softmax_anneal_ce_head(nn.Module):
    def __init__(self, backbone, model_settings):
        super(AAM_normfree_softmax_anneal_ce_head, self).__init__()
        self.th_step = 28000.0
        self.iter = 0.0
        self.max_m = 0.5
        self.model_settings = model_settings

        self.backbone = nn.DataParallel(backbone)
        self.metrics = AAMSoftmax_normfree(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], s=model_settings['s'], m=model_settings['m'])
        self.loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, x, y):
        self.iter += 1.0
        m = min(self.max_m, (self.iter / self.th_step) * self.model_settings['m'])

        emb = self.backbone(x, y)
        logits = self.metrics(emb, y, s=self.model_settings['s'], m=m)
        loss = self.loss(logits, y)

        return loss, logits, emb 


class A_softmax_ce_head(nn.Module):
    def __init__(self, backbone, model_settings):
        super(A_softmax_ce_head, self).__init__()
        self.backbone = nn.DataParallel(backbone)
        self.metrics = ASoftmax(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], m=model_settings['m'], phiflag=True)
        self.loss = AngleLoss()
    
    def forward(self, x, y):
        emb = self.backbone(x, y)
        logits = self.metrics(emb)
        loss = self.loss(logits, y)

        return loss, logits[0], emb 
