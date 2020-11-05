# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2020-02-28)

# Reference: torchvision.models.resnet.py
# The torchvision.models.resnet.py is modified to apply resnet in xvector framework.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .large_margin_clf import *
from .att_model import *

__all__ = ["ResNet_ASV", "ResNet_ASV_Transformer", "ResNet_ASV_50", "ResNet_ASV_mrmh_att", "Resnet_ASV_large_margin_annealing", "Resnet_ASV_large_margin_annealing_labsm"]

def conv3x3(in_planes, out_planes, Conv=nn.Conv2d, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return Conv(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=True, dilation=dilation)


def conv1x1(in_planes, out_planes, Conv=nn.Conv2d, stride=1):
    """1x1 convolution"""
    return Conv(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, Conv=nn.Conv2d, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, norm_layer_params={}, full_pre_activation=True):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.downsample = downsample
        self.stride = stride
        self.full_pre_activation = full_pre_activation

        if self.full_pre_activation:
            self._full_pre_activation(inplanes, planes, Conv, stride, norm_layer, norm_layer_params)
        else:
            self._original(inplanes, planes, Conv, stride, norm_layer, norm_layer_params)

    def _original(self, inplanes, planes, Conv, stride, norm_layer, norm_layer_params):
        self.conv1 = conv3x3(inplanes, planes, Conv, stride)
        self.bn1 = norm_layer(planes, **norm_layer_params)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, Conv)
        self.bn2 = norm_layer(planes, **norm_layer_params)
        self.relu2 = nn.ReLU(inplace=True)

    def _full_pre_activation(self, inplanes, planes, Conv, stride, norm_layer, norm_layer_params):
        self.bn1 = norm_layer(inplanes, **norm_layer_params)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, Conv, stride)
        self.bn2 = norm_layer(planes, **norm_layer_params)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, Conv)

    def _original_forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out

    def _full_pre_activation_forward(self, x):
        """Reference: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity mappings in deep residual 
                      networks. Paper presented at the European conference on computer vision.
        """
        identity = x

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out

    def forward(self, x):
        if self.full_pre_activation:
            return self._full_pre_activation_forward(x)
        else:
            return self._original_forward(x)


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, Conv=nn.Conv2d, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, norm_layer_params={}, full_pre_activation=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        
        self.downsample = downsample
        self.stride = stride
        self.full_pre_activation = full_pre_activation

        if self.full_pre_activation:
            self._full_pre_activation(inplanes, planes, Conv, groups, width, stride, norm_layer, norm_layer_params, dilation)
        else:
            self._original(inplanes, planes, Conv, groups, width, stride, norm_layer, norm_layer_params, dilation)

    def _original(self, inplanes, planes, Conv, groups, width, stride, norm_layer, norm_layer_params, dilation):
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, Conv)
        self.bn1 = norm_layer(width, **norm_layer_params)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(width, width, Conv, stride, groups, dilation)
        self.bn2 = norm_layer(width, **norm_layer_params)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(width, planes * self.expansion, Conv)
        self.bn3 = norm_layer(planes * self.expansion, **norm_layer_params)
        self.relu3 = nn.ReLU(inplace=True)

    def _full_pre_activation(self, inplanes, planes, Conv, groups, width, stride, norm_layer, norm_layer_params, dilation):
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.bn1 = norm_layer(inplanes, **norm_layer_params)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv1x1(inplanes, width, Conv)
        self.bn2 = norm_layer(width, **norm_layer_params)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(width, width, Conv, stride, groups, dilation)
        self.bn3 = norm_layer(width, **norm_layer_params)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(width, planes * self.expansion, Conv)

    def _original_forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out

    def _full_pre_activation_forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out

    def forward(self, x):
        if self.full_pre_activation:
            return self._full_pre_activation_forward(x)
        else:
            return self._original_forward(x)



class ResNet(nn.Module):
    """Just return a structure (preconv + resnet) without avgpool and final linear.
    """
    def __init__(self, head_inplanes, block="BasicBlock", layers=[3, 4, 6, 3], planes=[32, 64, 128, 256], convXd=2, 
                 full_pre_activation=True,
                 head_conv=True, head_conv_params={"kernel_size":3, "stride":1, "padding":1},
                 head_maxpool=False, head_maxpool_params={"kernel_size":3, "stride":1, "padding":1},
                 zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, norm_layer_params={}, downsample_type="resnet-type-d"):
        super(ResNet, self).__init__()

        if convXd != 1 and convXd != 2:
            raise TypeError("Expected 1d or 2d conv, but got {}.".format(convXd))

        if norm_layer is None:
            if convXd == 2:
                norm_layer = nn.BatchNorm2d
            else:
                norm_layer = nn.BatchNorm1d

        self._norm_layer = norm_layer

        self.inplanes = planes[0]
        if not head_conv and self.in_planes != head_inplanes:
            raise ValueError("The inplanes is not equal to resnet first block" \
                             "inplanes without head conv({} vs. {}).".format(head_inplanes, self.inplanes))
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        if block == "BasicBlock":
            used_block = BasicBlock
        elif block == "Bottleneck":
            used_block = Bottleneck
        else:
            raise TypeError("Do not support {} block.".format(block))
    
        self.groups = groups
        self.base_width = width_per_group
        self.head_conv = head_conv
        self.head_maxpool = head_maxpool

        self.downsample_multiple = 1
        self.full_pre_activation = full_pre_activation
        self.norm_layer_params = norm_layer_params

        self.Conv = nn.Conv2d if convXd == 2 else nn.Conv1d

        self.downsample_type = downsample_type

        if self.head_conv:
            # Keep conv1.outplanes == layer1.inplanes
            self.conv1 = self.Conv(head_inplanes, self.inplanes, **head_conv_params, bias=True)
            self.bn1 = norm_layer(self.inplanes, **norm_layer_params)
            self.relu = nn.ReLU(inplace=True)
            self.downsample_multiple *= head_conv_params["stride"]

        if self.head_maxpool:
            Maxpool = nn.MaxPool2d if convXd == 2 else nn.MaxPool1d
            self.maxpool = Maxpool(**head_maxpool_params)
            self.downsample_multiple *= head_maxpool_params["stride"]

        self.layer1 = self._make_layer(used_block, planes[0], layers[0])
        self.layer2 = self._make_layer(used_block, planes[1], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(used_block, planes[2], layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(used_block, planes[3], layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.downsample_multiple *= 8
        self.output_planes = planes[3] * used_block.expansion

        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        if "affine" in norm_layer_params.keys():
            norm_layer_affine = norm_layer_params["affine"]
        else:
            norm_layer_affine = True # torch.nn default it True

        for m in self.modules():
            if isinstance(m, self.Conv):
                # torch.nn.init.normal_(m.weight, 0., 0.01)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)) and norm_layer_affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual and norm_layer_affine:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0.0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0.0)

    def get_downsample_multiple(self):
        return self.downsample_multiple

    def get_output_planes(self):
        return self.output_planes

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:

            # resnet type b c [d] ref:https://arxiv.org/pdf/1812.01187v2.pdf
            if self.downsample_type == 'ori':
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, self.Conv, stride),
                    norm_layer(planes * block.expansion, **self.norm_layer_params),
                )
            elif self.downsample_type == "resnet-type-d":
                # downsample = nn.Sequential(
                #     torch.nn.AvgPool2d(kernel_size=[3, 3], stride=2, padding=1),
                #     conv1x1(self.inplanes, planes * block.expansion, self.Conv, stride=1),
                #     norm_layer(planes * block.expansion, **self.norm_layer_params),
                # )

                downsample = nn.Sequential(
                    torch.nn.AvgPool2d(kernel_size=[3, 3], stride=stride, padding=1),
                    conv1x1(self.inplanes, planes * block.expansion, self.Conv, stride=1),
                    norm_layer(planes * block.expansion, **self.norm_layer_params),
                )

        layers = []
        layers.append(block(self.inplanes, planes, self.Conv, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, 
                            norm_layer_params=self.norm_layer_params,
                            full_pre_activation=self.full_pre_activation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.Conv, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, norm_layer_params=self.norm_layer_params,
                                full_pre_activation=self.full_pre_activation))

        return nn.Sequential(*layers)


    def forward(self, x): 
        
        if self.head_conv:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        
        if self.head_maxpool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class ResNet_ASV(nn.Module):
    def __init__(self, utt_dim=1280, embedding_dim=256):
        super(ResNet_ASV, self).__init__()
        self.backbone = ResNet(1)

        self.utt_dim = utt_dim
        self.embedding_dim = embedding_dim

        self.embedding_layer1 = torch.nn.Sequential()
        self.embedding_layer1.add_module('linear', nn.Linear(self.utt_dim*2, self.embedding_dim))
        self.embedding_layer1.add_module('relu', nn.ReLU(True))
        self.embedding_layer1.add_module('batchnorm',nn.BatchNorm1d(self.embedding_dim))

        nn.init.kaiming_normal_(self.embedding_layer1.linear.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.embedding_layer1.linear.bias, 0.0)
        nn.init.constant_(self.embedding_layer1.batchnorm.weight, 1.0)
        nn.init.constant_(self.embedding_layer1.batchnorm.bias, 0.0)


    def stat_pool(self, stat_src):
        stat_mean = torch.mean(stat_src,dim=2)
        stat_std = torch.sqrt(torch.var(stat_src,dim=2)+0.00001)
        stat_pool_out = torch.cat((stat_mean,stat_std),1)
        return stat_pool_out

    def forward(self, x, y):
        x = x.permute(0,2,1)
        x = x.unsqueeze(1)
        x = self.backbone(x)
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3])
        x = self.stat_pool(x)
        out = self.embedding_layer1(x)

        return out

class ResNet_ASV_mrmh_att(nn.Module):
    def __init__(self, utt_dim=1280, embedding_dim=256):
        super(ResNet_ASV_mrmh_att, self).__init__()
        self.backbone = ResNet(1)

        self.utt_dim = utt_dim
        self.embedding_dim = embedding_dim

        self.embedding_layer1 = torch.nn.Sequential()
        self.embedding_layer1.add_module('linear', nn.Linear(self.utt_dim * 5, self.embedding_dim))
        self.embedding_layer1.add_module('relu', nn.ReLU(True))
        self.embedding_layer1.add_module('batchnorm',nn.BatchNorm1d(self.embedding_dim))

        nn.init.kaiming_normal_(self.embedding_layer1.linear.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.embedding_layer1.linear.bias, 0.0)
        nn.init.constant_(self.embedding_layer1.batchnorm.weight, 1.0)
        nn.init.constant_(self.embedding_layer1.batchnorm.bias, 0.0)

        self.embedding_layer2 = torch.nn.Sequential()
        self.embedding_layer2.add_module('linear', nn.Linear(self.embedding_dim, self.embedding_dim))
        self.embedding_layer2.add_module('relu', nn.ReLU(True))
        self.embedding_layer2.add_module('batchnorm',nn.BatchNorm1d(self.embedding_dim))

        nn.init.kaiming_normal_(self.embedding_layer2.linear.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.embedding_layer2.linear.bias, 0.0)
        nn.init.constant_(self.embedding_layer2.batchnorm.weight, 1.0)
        nn.init.constant_(self.embedding_layer2.batchnorm.bias, 0.0)

        self.mrmh_att = MRMH_att_module(utt_dim=self.utt_dim)

    def stat_pool(self, stat_src):
        stat_mean = torch.mean(stat_src,dim=2)
        stat_std = torch.sqrt(torch.var(stat_src,dim=2)+0.00001)
        stat_pool_out = torch.cat((stat_mean,stat_std),1)
        return stat_pool_out

    def forward(self, x, y):
        x = x.permute(0,2,1)
        x = x.unsqueeze(1)
        x = self.backbone(x)
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3])
        x = self.mrmh_att(x)

        out = self.embedding_layer1(x)
        out = self.embedding_layer2(out)

        return out


class ResNet_ASV_50(nn.Module):
    def __init__(self, utt_dim=1280*4, embedding_dim=256):
        super(ResNet_ASV_50, self).__init__()
        self.backbone = ResNet(1, block="Bottleneck")

        self.utt_dim = utt_dim
        self.embedding_dim = embedding_dim

        self.embedding_layer1 = torch.nn.Sequential()
        self.embedding_layer1.add_module('linear', nn.Linear(self.utt_dim*2, self.embedding_dim))
        self.embedding_layer1.add_module('relu', nn.ReLU(True))
        self.embedding_layer1.add_module('batchnorm',nn.BatchNorm1d(self.embedding_dim))

        nn.init.kaiming_normal_(self.embedding_layer1.linear.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.embedding_layer1.linear.bias, 0.0)
        nn.init.constant_(self.embedding_layer1.batchnorm.weight, 1.0)
        nn.init.constant_(self.embedding_layer1.batchnorm.bias, 0.0)


    def stat_pool(self, stat_src):
        stat_mean = torch.mean(stat_src,dim=2)
        stat_std = torch.sqrt(torch.var(stat_src,dim=2)+0.00001)
        stat_pool_out = torch.cat((stat_mean,stat_std),1)
        return stat_pool_out

    def forward(self, x, y):
        x = x.permute(0,2,1)
        x = x.unsqueeze(1)
        x = self.backbone(x)
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3])
        x = self.stat_pool(x)

        out = self.embedding_layer1(x)

        return out


class ResNet_ASV_Transformer(nn.Module):
    def __init__(self, utt_dim=1280, embedding_dim=256):
        super(ResNet_ASV_Transformer, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=40, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        
        self.backbone = ResNet(2)

        self.utt_dim = utt_dim
        self.embedding_dim = embedding_dim

        self.embedding_layer1 = torch.nn.Sequential()
        self.embedding_layer1.add_module('linear', nn.Linear(self.utt_dim*2, self.embedding_dim))
        self.embedding_layer1.add_module('relu', nn.ReLU(True))
        self.embedding_layer1.add_module('batchnorm',nn.BatchNorm1d(self.embedding_dim))

        nn.init.kaiming_normal_(self.embedding_layer1.linear.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.embedding_layer1.linear.bias, 0.0)
        nn.init.constant_(self.embedding_layer1.batchnorm.weight, 1.0)
        nn.init.constant_(self.embedding_layer1.batchnorm.bias, 0.0)


    def stat_pool(self, stat_src):
        stat_mean = torch.mean(stat_src,dim=2)
        stat_std = torch.sqrt(torch.var(stat_src,dim=2)+0.00001)
        stat_pool_out = torch.cat((stat_mean,stat_std),1)
        return stat_pool_out

    def forward(self, x, y):
        x_1 = x.permute(1, 0, 2)
        out = self.transformer_encoder(x_1)
        out = out.permute(1, 0, 2)

        x = x.permute(0, 2, 1)
        out = out.permute(0, 2, 1)

        x = x.unsqueeze(1)
        out = out.unsqueeze(1)

        x_ready = torch.cat([x, out], dim=1)

        # x = x.permute(0,2,1)
        # x = x.unsqueeze(1)
        x = self.backbone(x_ready)
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3])
        x = self.stat_pool(x)
        out = self.embedding_layer1(x)

        return out

class Resnet_ASV_large_margin_annealing(nn.Module):
    def __init__(self, backbone, model_settings):
        super(Resnet_ASV_large_margin_annealing, self).__init__()
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

        emb2 = self.backbone(x, y)
        logits, nm_W = self.metrics(emb2, y, s=self.model_settings['s'], m=m)
        loss = self.loss(logits, y)

        pred = logits.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        label = y.data.cpu().numpy()
        acc = np.mean((pred == label).astype(int))

        return loss, logits, emb2, acc, 0.0  

class Resnet_ASV_large_margin_annealing_labsm(nn.Module):
    def __init__(self, backbone, model_settings):
        super(Resnet_ASV_large_margin_annealing_labsm, self).__init__()
        self.th_step = model_settings['anneal_steps']
        self.iter = 0.0
        self.max_m = model_settings['m']
        self.model_settings = model_settings
        
        self.backbone = nn.DataParallel(backbone)
        self.metrics = AMSoftmax_normfree(in_features=model_settings['emb_size'], out_features=model_settings['class_num'], s=model_settings['s'], m=model_settings['m'])
        # self.loss = torch.nn.CrossEntropyLoss()
        self.LogSoftmax = torch.nn.LogSoftmax(dim=1)
        eps_labsm = 0.1
        self.max_labsm = (1 - eps_labsm)
        self.other_labsm = eps_labsm / (model_settings['class_num'] - 1)
    
    def forward(self, x, y, mod):
        if mod == 'train':
            self.iter += 1.0
            m = min(self.max_m, (self.iter / self.th_step) * self.model_settings['m'])
        else:
            m = 0.0

        emb2 = self.backbone(x, y)
        logits, nm_W = self.metrics(emb2, y, s=self.model_settings['s'], m=m)

        logsoft = self.LogSoftmax(logits)
        labsm_mask = torch.ones(logsoft.size()).cuda() * self.other_labsm
        for i, j in zip(labsm_mask, y):
            i[j] = self.max_labsm
        loss = - torch.sum(logsoft*labsm_mask) / y.size(0)

        pred = logits.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        label = y.data.cpu().numpy()
        acc = np.mean((pred == label).astype(int))

        return loss, logits, emb2, acc, 0.0  