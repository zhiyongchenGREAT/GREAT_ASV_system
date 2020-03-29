import torch
import torch.nn as nn
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
        
        # self.downsampler = nn.Sequential(
        #     nn.BatchNorm2d(input_dim),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=1, stride=2, padding=0)
        # )
        self.downsampler = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=1, stride=2, padding=0)

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


class PreactivateBottleneckBlock(nn.Module):
    def __init__(self, input_dim, output_dim, downsample=False, firstact=True):
        super(PreactivateBottleneckBlock, self).__init__()
        self.downsample = downsample
        self.firstact = firstact
        
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

        out = self.bn3(out)
        out = self.relu3(out)        
        out = self.conv3(out)

        if self.downsample is True:
            residual = self.downsampler(x)

        out += residual
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
                    # self.resblocks.append(PreactivateBlock(group_dims[group_i], group_dims[group_i], downsample=True, firstact=False))
                    self.resblocks.append(PreactivateBlock(group_dims[group_i], group_dims[group_i], downsample=False, firstact=False))
                elif group_i != 0 and block_i == 0:
                    self.resblocks.append(PreactivateBlock(group_dims[group_i-1], group_dims[group_i], downsample=True, firstact=True))
                else:
                    raise Exception("Wrong building resblocks")
        
        self.fc1 = nn.Sequential(
            nn.BatchNorm2d(group_dims[-1]),
            nn.ReLU(),
            # nn.Conv2d(group_dims[-1], out_channels=512, kernel_size=(8, 1), stride=1, padding=0),
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
        
        out = self.fc1(out)
        out = self.pooltime(out)
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
                    self.resblocks.append(PreactivateBottleneckBlock(group_dims[group_i], group_dims[group_i], downsample=False, firstact=True))
                elif group_i == 0 and block_i == 0:
                    self.resblocks.append(PreactivateBottleneckBlock(64, group_dims[group_i], downsample=True, firstact=False))
                elif group_i != 0 and block_i == 0:
                    self.resblocks.append(PreactivateBottleneckBlock(group_dims[group_i-1], group_dims[group_i], downsample=True, firstact=True))
                else:
                    raise Exception("Wrong building resblocks")
        
        self.fc1 = nn.Sequential(
            nn.BatchNorm2d(group_dims[-1]),
            nn.ReLU(),
            nn.Conv2d(group_dims[-1], out_channels=2048, kernel_size=(8, 1), stride=1, padding=0),
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
        
        out = self.fc1(out)
        out = self.pooltime(out)
        out = out.squeeze(-1)
        out = out.squeeze(-1)
        return out


class Linear_softmax_ce_head(nn.Module):
    def __init__(self, backbone, model_settings):
        super(Linear_softmax_ce_head, self).__init__()
        self.backbone = backbone
        self.linear = nn.Linear(model_settings['emb_size'], model_settings['class_num'])
        self.loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, x, y):
        emb = self.backbone(x, y)
        logits = self.linear(emb)
        loss = self.loss(logits, y)

        return loss, logits, emb       

