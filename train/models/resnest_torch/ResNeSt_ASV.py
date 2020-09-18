import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
import torch
import torch.nn as nn
import numpy as np
from .resnet import Bottleneck, Bottleneck_Lite
from large_margin_clf import *



__all__ = ['ResNeSt_ASV', 'ResNeSt_Lite_ASV', 'ResNeSt_ASV_large_margin_annealing']

# def resnest50(pretrained=False, root='~/.encoding/models', **kwargs):
#     model = ResNet(Bottleneck, [3, 4, 6, 3],
#                    radix=2, groups=1, bottleneck_width=64,
#                    deep_stem=True, stem_width=32, avg_down=True,
#                    avd=True, avd_first=False, **kwargs)
#     if pretrained:
#         model.load_state_dict(torch.hub.load_state_dict_from_url(
#             resnest_model_urls['resnest50'], progress=True, check_hash=True))
#     return model

def conv3x3(in_planes, out_planes, Conv=nn.Conv2d, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return Conv(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=True, dilation=dilation)


def conv1x1(in_planes, out_planes, Conv=nn.Conv2d, stride=1):
    """1x1 convolution"""
    return Conv(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)


class ResNeSt(nn.Module):
    """Just return a structure (preconv + resnet) without avgpool and final linear.
    """
    def __init__(self, head_inplanes, block="Bottleneck_Lite", layers=[3, 4, 6, 3], planes=[32, 64, 128, 256], convXd=2, 
                 full_pre_activation=True,
                 head_conv=True, head_conv_params={"kernel_size":3, "stride":1, "padding":1},
                 head_maxpool=False, head_maxpool_params={"kernel_size":3, "stride":1, "padding":1},
                 zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, norm_layer_params={}, downsample_type="resnet-type-d"):
        super(ResNeSt, self).__init__()

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

        if block == "Bottleneck":
            used_block = Bottleneck
        elif block == "Bottleneck_Lite":
            used_block = Bottleneck_Lite
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
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=2, cardinality=1,
                                bottleneck_width=64,
                                avd=True, avd_first=False,
                                dilation=1, is_first=False, rectified_conv=False,
                                rectify_avg=False,
                                norm_layer=nn.BatchNorm2d, dropblock_prob=0.0,
                                last_gamma=False))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                radix=2, cardinality=1,
                                bottleneck_width=64,
                                avd=True, avd_first=False,
                                dilation=1, rectified_conv=False,
                                rectify_avg=False,
                                norm_layer=nn.BatchNorm2d, dropblock_prob=0.0,
                                last_gamma=False))  

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


class ResNeSt_Lite_ASV(nn.Module):
    def __init__(self, utt_dim=1280, embedding_dim=256):
        super(ResNeSt_Lite_ASV, self).__init__()
        self.backbone = ResNeSt(1, block="Bottleneck_Lite")

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

class ResNeSt_ASV(nn.Module):
    def __init__(self, utt_dim=1280*4, embedding_dim=256):
        super(ResNeSt_ASV, self).__init__()
        self.backbone = ResNeSt(1, block="Bottleneck")

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

class ResNeSt_ASV_large_margin_annealing(nn.Module):
    def __init__(self, backbone, model_settings):
        super(ResNeSt_ASV_large_margin_annealing, self).__init__()
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