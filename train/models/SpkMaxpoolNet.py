import torch
import torch.nn as nn
from models.metrics import *

class SpeakerMaxpoolNet_nohead(torch.nn.Module):
    def __init__(self, output_dim):
        super(SpeakerMaxpoolNet_nohead, self).__init__()
        self.in_channels = [23, 128, 128, 128]
        self.layer_sizes = [256, 256, 256, 2048]
        self.kernel_sizes = [7, 5, 3, 1]

        self.stats_dim = 1024
        self.stats_out_dim = 2048
        self.embedding_dim1 = 1024
        self.embedding_dim2 = 512
        self.output_dim = output_dim

        self.frame1 = torch.nn.Sequential(
            nn.Conv1d(self.in_channels[0],self.layer_sizes[0],self.kernel_sizes[0]),
            nn.PReLU(),
            nn.BatchNorm1d(self.layer_sizes[0])
        )

        self.maxpool1 = torch.nn.MaxPool2d((2, 2))

        self.frame2 = torch.nn.Sequential(
            nn.Conv1d(self.in_channels[1],self.layer_sizes[1],self.kernel_sizes[1]),
            nn.PReLU(),
            nn.BatchNorm1d(self.layer_sizes[1])
        )

        self.maxpool2 = torch.nn.MaxPool2d((2, 2))

        self.frame3 = torch.nn.Sequential(
            nn.Conv1d(self.in_channels[2],self.layer_sizes[2],self.kernel_sizes[2]),
            nn.PReLU(),
            nn.BatchNorm1d(self.layer_sizes[2])
        )

        self.maxpool3 = torch.nn.MaxPool2d((2, 2))

        self.frame4 = torch.nn.Sequential(
            nn.Conv1d(self.in_channels[3],self.layer_sizes[3],self.kernel_sizes[3]),
            nn.PReLU(),
            nn.BatchNorm1d(self.layer_sizes[3])
        )

        self.maxpool4 = torch.nn.MaxPool2d((2, 2))        
        
        self.embedding_layer1 = torch.nn.Sequential()
        self.embedding_layer1.add_module('linear', nn.Linear(self.stats_out_dim, self.embedding_dim1))
        self.embedding_layer1.add_module('prelu', nn.PReLU())
        self.embedding_layer1.add_module('batchnorm', nn.BatchNorm1d(self.embedding_dim1))
        
        self.embedding_layer2 = torch.nn.Sequential()
        self.embedding_layer2.add_module('linear', nn.Linear(self.embedding_dim1, self.embedding_dim2))
        self.embedding_layer2.add_module('prelu', nn.PReLU())
        self.embedding_layer2.add_module('batchnorm', nn.BatchNorm1d(self.embedding_dim2))


    def stat_pool(self, stat_src):
        stat_mean = torch.mean(stat_src,dim=2)
        stat_std = torch.sqrt(torch.var(stat_src,dim=2)+0.00001)
        stat_pool_out = torch.cat((stat_mean,stat_std),1)
        return stat_pool_out

    def forward(self, x, y):
        src = x.permute(0,2,1)
        out = self.frame1(src)
        out = self.maxpool1(out)
        out = self.frame2(out)
        out = self.maxpool2(out)
        out = self.frame3(out)
        out = self.maxpool3(out)
        out = self.frame4(out)
        out = self.maxpool4(out)
        out = self.stat_pool(out)
        out_embedding1 = self.embedding_layer1(out)
        out_embedding2 = self.embedding_layer2(out_embedding1)

        return out_embedding2

class SpeakerMaxpoolNet_simple_head(torch.nn.Module):
    def __init__(self, output_dim, model_settings):
        super(SpeakerMaxpoolNet_simple_head, self).__init__()
        self.backbone = SpeakerMaxpoolNet_nohead(output_dim)
        self.head = nn.Sequential()
        self.head.add_module('linear', nn.Linear(model_settings['emb_size'], output_dim))
    
    def forward(self, x, y):
        out = self.backbone(x, y)
        out = self.head(out)
        return out

class SpeakerMaxpoolNet_arc_margin(torch.nn.Module):
    def __init__(self, output_dim, model_settings):
        super(SpeakerMaxpoolNet_arc_margin, self).__init__()
        self.backbone = SpeakerMaxpoolNet_nohead(output_dim)
        self.head = ArcMarginProduct(model_settings['emb_size'], output_dim, s=model_settings['scale'], m=model_settings['margin'])
    def forward(self, x, y):
        out = self.backbone(x, y)
        out = self.head(out, y)
        return out

