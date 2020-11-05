import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MRMH_att_module"]

class MRMH_att_module(nn.Module):
    def __init__(self, utt_dim):
        super(MRMH_att_module, self).__init__()
        self.utt_dim = utt_dim

        self.head1 = torch.nn.Sequential()
        self.head1.add_module('linear1', nn.Linear(self.utt_dim, self.utt_dim, bias=True))
        self.head1.add_module('relu', nn.ReLU(True))
        self.head1.add_module('linear2', nn.Linear(self.utt_dim, 1, bias=True))

        nn.init.kaiming_uniform_(self.head1.linear1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.head1.linear1.bias, 0.0)
        # nn.init.kaiming_uniform_(self.head1.linear2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.head1.linear2.bias, 0.0)

        self.head2 = torch.nn.Sequential()
        self.head2.add_module('linear1', nn.Linear(self.utt_dim, self.utt_dim, bias=True))
        self.head2.add_module('relu', nn.ReLU(True))
        self.head2.add_module('linear2', nn.Linear(self.utt_dim, 1, bias=True))

        nn.init.kaiming_uniform_(self.head2.linear1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.head2.linear1.bias, 0.0)
        # nn.init.kaiming_uniform_(self.head2.linear2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.head2.linear2.bias, 0.0)

        self.head3 = torch.nn.Sequential()
        self.head3.add_module('linear1', nn.Linear(self.utt_dim, self.utt_dim, bias=True))
        self.head3.add_module('relu', nn.ReLU(True))
        self.head3.add_module('linear2', nn.Linear(self.utt_dim, 1, bias=True))

        nn.init.kaiming_uniform_(self.head3.linear1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.head3.linear1.bias, 0.0)
        # nn.init.kaiming_uniform_(self.head3.linear2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.head3.linear2.bias, 0.0)

        self.head4 = torch.nn.Sequential()
        self.head4.add_module('linear1', nn.Linear(self.utt_dim, self.utt_dim, bias=True))
        self.head4.add_module('relu', nn.ReLU(True))
        self.head4.add_module('linear2', nn.Linear(self.utt_dim, 1, bias=True))

        nn.init.kaiming_uniform_(self.head4.linear1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.head4.linear1.bias, 0.0)
        # nn.init.kaiming_uniform_(self.head4.linear2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.head4.linear2.bias, 0.0)

        self.head5 = torch.nn.Sequential()
        self.head5.add_module('linear1', nn.Linear(self.utt_dim, self.utt_dim, bias=True))
        self.head5.add_module('relu', nn.ReLU(True))
        self.head5.add_module('linear2', nn.Linear(self.utt_dim, 1, bias=True))

        nn.init.kaiming_uniform_(self.head5.linear1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.head5.linear1.bias, 0.0)
        # nn.init.kaiming_uniform_(self.head5.linear2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.head5.linear2.bias, 0.0)

        self.softmax = nn.Softmax(dim = 1)

    def forward(self, emb):

        # [B, C(1280 -- 5 * 256), T]
        # print(emb.shape)
        x = emb.permute(0,2,1)

        x1 = self.head1(x)/1.0
        x2 = self.head2(x)/1.0
        x3 = self.head3(x)/5.0
        x4 = self.head4(x)/5.0
        x5 = self.head5(x)/10.0

        x1 = self.softmax(x1) * x
        x2 = self.softmax(x2) * x
        x3 = self.softmax(x3) * x
        x4 = self.softmax(x4) * x
        x5 = self.softmax(x5) * x

        x1 = torch.sum(x1, axis=1)
        x2 = torch.sum(x2, axis=1)
        x3 = torch.sum(x3, axis=1)
        x4 = torch.sum(x4, axis=1)
        x5 = torch.sum(x5, axis=1)

        # [B, 1280]

        # out = torch.cat((x1, x2, x3, x4, x5), dim=1)
        out = torch.cat((x1, x2, x3, x4, x5), dim=1)

        return out
        # [B, 1280 * 5]
