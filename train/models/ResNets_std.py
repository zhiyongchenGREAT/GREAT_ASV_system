import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from models.metrics import *

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


class PreactivateBottleneckBlock(nn.Module):
    def __init__(self, input_dim, output_dim, downsample=False, firstact=True, project=False):
        super(PreactivateBottleneckBlock, self).__init__()
        self.downsample = downsample
        self.firstact = firstact
        self.project = project
        
        reduced_dim = int(output_dim / 4)
        self.bn1 = nn.BatchNorm2d(input_dim)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=reduced_dim, kernel_size=1, stride=1, padding=0)

        self.bn2 = nn.BatchNorm2d(reduced_dim)
        self.relu2 = nn.ReLU()
        if downsample is True:
            self.conv2 = nn.Conv2d(in_channels=reduced_dim, out_channels=reduced_dim, kernel_size=3, stride=2, padding=1)
        elif downsample is False:
            self.conv2 = nn.Conv2d(in_channels=reduced_dim, out_channels=reduced_dim, kernel_size=3, stride=1, padding=1)
        else:
            raise Exception('downsample wrong setting')

        self.bn3 = nn.BatchNorm2d(reduced_dim)
        self.relu3 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=reduced_dim, out_channels=output_dim, kernel_size=1, stride=1, padding=0)
        
        self.downsampler = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=1, stride=2, padding=0)
        )

        self.projector = nn.Sequential(
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

        out = self.bn3(out)
        out = self.relu3(out)        
        out = self.conv3(out)

        if self.downsample is True:
            residual = self.downsampler(downsample_out)
        
        if self.project is True:
            residual = self.projector(x)

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
#             # print(out.size())
        
#         out = self.fc1(out)
#         out = self.pooltime(out)
#         out = out.squeeze(-1)
#         out = out.squeeze(-1)
#         return out

class Resnet18(nn.Module):
    def __init__(self, feat_dim, emb_dim):
        super(Resnet18, self).__init__()
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
            # print(out.size())
        
        # out = self.fc1(out)
        # print(out.size())
        out = self.pooltime(out)
        # print(out.size())
        out = out.squeeze(-1)
        out = out.squeeze(-1)
        return out


class Resnet34(nn.Module):
    def __init__(self, feat_dim, emb_dim):
        super(Resnet34, self).__init__()
        group_nums = [3, 4, 6, 3]
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


class Resnet50(nn.Module):
    def __init__(self, feat_dim, emb_dim):
        super(Resnet50, self).__init__()
        group_nums = [3, 4, 6, 3]
        group_dims = [256, 512, 1024, 2048]
        self.resblocks = nn.ModuleList([])

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)

        for group_i, group_num in enumerate(group_nums):
            for block_i in range(group_num):
                if block_i != 0:
                    self.resblocks.append(PreactivateBottleneckBlock(group_dims[group_i], group_dims[group_i], downsample=False, firstact=True, project=False))
                elif group_i == 0 and block_i == 0:
                    self.resblocks.append(PreactivateBottleneckBlock(64, group_dims[group_i], downsample=False, firstact=False, project=True))
                elif group_i != 0 and block_i == 0:
                    self.resblocks.append(PreactivateBottleneckBlock(group_dims[group_i-1], group_dims[group_i], downsample=True, firstact=True, project=False))
                else:
                    raise Exception("Wrong building resblocks")
        
        self.fc1 = nn.Sequential(
            nn.BatchNorm2d(group_dims[-1]),
            nn.ReLU(),
            nn.Conv2d(group_dims[-1], out_channels=2048, kernel_size=(16, 1), stride=1, padding=0),
            nn.BatchNorm2d(2048)
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

        emb2 = self.backbone(x, y)
        logits, nm_W = self.metrics(emb2, y, s=self.model_settings['s'], m=m)
        loss = self.loss(logits, y)

        pred = logits.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        label = y.data.cpu().numpy()
        acc = np.mean((pred == label).astype(int))

        return loss, logits, emb2, acc, 0.0  

