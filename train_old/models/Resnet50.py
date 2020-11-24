import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_planes = 32

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        # self.linear = nn.Linear(512*block.expansion, num_classes)

        self.embedding_layer1 = torch.nn.Sequential()
        self.embedding_layer1.add_module('linear', nn.Linear(256*2, 256))
        # self.embedding_layer1.add_module('relu', nn.ReLU(True))
        self.embedding_layer1.add_module('batchnorm',nn.BatchNorm1d(256))
    
    def stat_pool(self, stat_src):
        stat_mean = torch.mean(stat_src,dim=[2, 3])
        stat_std = torch.sqrt(torch.var(stat_src,dim=[2, 3])+0.00001)
        stat_pool_out = torch.cat((stat_mean,stat_std),1)
        return stat_pool_out  

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, y):
        out = x.permute(0,2,1)
        out = out.unsqueeze(1)

        out = F.relu(self.bn1(self.conv1(out)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # print(out.shape)
        out = self.stat_pool(out)
        out = out.squeeze(-1)
        out = out.squeeze(-1)
        # out = F.avg_pool2d(out, 4)
        # out = out.view(out.size(0), -1)
        out = self.embedding_layer1(out)

        return out



def ResNet50_SAP_T(feat_dim=None, emb_dim=None):
    return ResNet(Bottleneck, [3,4,6,3])

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