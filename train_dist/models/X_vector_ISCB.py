import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import PreEmphasis
import torchaudio
import os
from .inv_specaug import SpecAugment

class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, bottleneck_dim):
        super().__init__()
        # Use Conv1d with stride == 1 rather than Linear, then we don't need to transpose inputs.
        self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1) # equals W and b in the paper
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1) # equals V and k in the paper

    def forward(self, x):
        # DON'T use ReLU here! In experiments, I find ReLU hard to converge.
        alpha = torch.tanh(self.linear1(x))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * x ** 2, dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)

class Xvector_1L_ISCB(torch.nn.Module):
    def __init__(self, in_channels, embd_dim, spec_aug=False):
        super(Xvector_1L_ISCB, self).__init__()
        self.feature_dim = in_channels
        self.embedding_dim = embd_dim
        self.in_channels = [self.feature_dim, 512, 512, 512, 512]
        self.layer_sizes = [512, 512, 512, 512, 1500]
        self.kernel_sizes = [5, 3, 3, 1, 1]
        self.dilations = [1,2,3,1,1]

        self.instancenorm   = nn.InstanceNorm1d(in_channels)
        self.torchfb        = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, window_fn=torch.hamming_window, n_mels=in_channels)
                )
        self.spec_aug = spec_aug
        if self.spec_aug:
            self.spec_aug_f = SpecAugment(frequency=0.2, frame=0.0, rows=1, cols=1, random_rows=False, random_cols=False)

        print('%s, Embedding size is %d,  Spec_aug %s.'%(os.path.basename(__file__), embd_dim, str(self.spec_aug)))

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
       
        self.pooling = AttentiveStatsPool(self.layer_sizes[4], 128)

        self.embedding_layer1 = torch.nn.Sequential()
        self.embedding_layer1.add_module('linear', nn.Linear(self.layer_sizes[4]*2, self.embedding_dim))
        self.embedding_layer1.add_module('relu', nn.ReLU(True))
        self.embedding_layer1.add_module('batchnorm',nn.BatchNorm1d(self.embedding_dim))
        
        self.embedding_layer2 = torch.nn.Sequential()
        self.embedding_layer2.add_module('linear', nn.Linear(self.embedding_dim, self.embedding_dim))
        self.embedding_layer2.add_module('tanh', nn.Tanh())
        self.embedding_layer2.add_module('batchnorm',nn.BatchNorm1d(self.embedding_dim))
        
        self.embedding_layer3 = torch.nn.Sequential()
        self.embedding_layer3.add_module('linear', nn.Linear(self.embedding_dim, self.embedding_dim))
        self.embedding_layer3.add_module('tanh', nn.Tanh())
        self.embedding_layer3.add_module('batchnorm',nn.BatchNorm1d(self.embedding_dim))

        self.fc_layer1 = torch.nn.Sequential()
        self.fc_layer1.add_module('linear', nn.Linear(self.embedding_dim, 1))
        self.fc_layer1.add_module('relu', nn.ReLU(True))
        
        self.bn = nn.BatchNorm1d(self.embedding_dim)

    def forward(self, x):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.torchfb(x)+1e-6
                x = x.log()
                x = self.instancenorm(x)
                if self.spec_aug and self.training:
                    for i in x:
                        _ = self.spec_aug_f(i)

        # src = x.permute(0,2,1)
        out = self.tdnn1(x)
        out = self.tdnn2(out)
        out = self.tdnn3(out)
        out = self.tdnn4(out)
        out = self.tdnn5(out)

        out = self.pooling(out)

        out_embedding1 = self.embedding_layer1(out)
        out_embedding2 = self.embedding_layer3(self.embedding_layer2(out_embedding1))
        out_f = self.fc_layer1(out_embedding1)
        out_embedding2 = self.bn(out_embedding2*out_f+out_embedding1)

        return [out_f, out_embedding2, out_embedding1]

def MainModel(n_mels, nOut, spec_aug, **kwargs):
    model = Xvector_1L_ISCB(in_channels=n_mels, embd_dim=nOut, spec_aug=spec_aug)
    return model
