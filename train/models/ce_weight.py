import torch
import torch.nn as nn


class Ce_weight(nn.Module):

    def __init__(self):
        super(Ce_weight, self).__init__()
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, predict, target, weight):
        ce1 = self.ce(predict[0], target)
        ce2 = self.ce(predict[1], target)
        loss =  weight * ce1 + (1 - weight) * ce2
        return loss