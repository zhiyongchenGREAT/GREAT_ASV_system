import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .inv_specaug import SpecAugment
from utils import PreEmphasis
import torchaudio

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

''' Attentive weighted mean and standard deviation pooling.
'''
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

class ViT(nn.Module):
    def __init__(self, training_frames=302, out_dim=128, inp_dim=40, dim=512, depth=12, heads=4, mlp_dim=1024, pool='mean', dim_head=64, dropout=0., emb_dropout=0., spec_aug=True):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            nn.Linear(inp_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, training_frames, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # ASV related
        self.asv_out = nn.Sequential(
            AttentiveStatsPool(dim, 128),
            nn.BatchNorm1d(dim * 2),
            nn.Linear(dim * 2, out_dim),
            nn.BatchNorm1d(out_dim)
        )

        self.spec_aug = spec_aug
        self.spec_aug_f = SpecAugment(frequency=0.2, frame=0.0, rows=1, cols=1, random_rows=False, random_cols=False)
        self.instancenorm   = nn.InstanceNorm1d(inp_dim)
        self.torchfb        = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, window_fn=torch.hamming_window, n_mels=inp_dim)
                )


    def forward(self, x):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.torchfb(x)+1e-6
                x = x.log()
                x = self.instancenorm(x)
                if self.spec_aug and self.training:
                    for i in x:
                        _ = self.spec_aug_f(i)
        # print(x.size())

        x = x.transpose(1, 2)
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)

        if self.training:
            x += self.pos_embedding[:, :(n)]
        else:
            testing_frames = n
            interpolate_pe = torch.nn.functional.upsample(self.pos_embedding.transpose(1, 2), size=[testing_frames], mode='linear', align_corners=True)
            interpolate_pe = interpolate_pe.transpose(1, 2)
            x += interpolate_pe[:, :(n)]
        
        x = self.dropout(x)
        x = self.transformer(x)

        x = x.transpose(1, 2)

        x = self.asv_out(x)

        return x

def MainModel(n_mels, nOut, spec_aug, max_frames, **kwargs):
    model = ViT(training_frames=302, out_dim=nOut, inp_dim=n_mels, dim=512, depth=6, \
            heads=4, mlp_dim=1024, pool='mean', dim_head=64, \
            dropout=0., emb_dropout=0., spec_aug=spec_aug)
    return model