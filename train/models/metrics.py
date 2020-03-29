from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable
import math

class AAMSoftmax(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=50.0, m=0.50):
        super(AAMSoftmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
    
    def _calculate_para(self, s, m):
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

    def forward(self, input, label, s, m):
        self._calculate_para(s, m)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        mod_cosine = torch.zeros_like(cosine)
        mod_cosine = mod_cosine + cosine
        for i in range(cosine.size(0)):
            lb = int(label[i])
            if cosine[i, lb] > 0:
                mod_cosine[i, lb] = cosine[i, lb] * self.cos_m - sine[i, lb] * self.sin_m
            else:
                continue
        return self.s * mod_cosine

class AAMSoftmax_m(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=50.0, m=0.50):
        super(AAMSoftmax_m, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
    
    def _calculate_para(self, s, m):
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

    def forward(self, input, label, s, m):
        self._calculate_para(s, m)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))       
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        mod_cosine = torch.zeros_like(cosine)
        mod_cosine = mod_cosine + cosine

        for i in range(cosine.size(0)):
            lb = int(label[i])
            if cosine[i, lb] > math.cos(math.pi - m):
                mod_cosine[i, lb] = cosine[i, lb] * self.cos_m - sine[i, lb] * self.sin_m
            else:
                mod_cosine[i, lb] = cosine[i, lb] * self.cos_m - math.sin(math.pi - m) * self.sin_m
        return self.s * mod_cosine

# class AAMSoftmax_normfree(nn.Module):
#     r"""Implement of large margin arc distance: :
#         Args:
#             in_features: size of each input sample
#             out_features: size of each output sample
#             s: norm of input feature
#             m: margin
#             cos(theta + m)
#         """
#     def __init__(self, in_features, out_features, s=50.0, m=0.50):
#         super(AAMSoftmax_normfree, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features

#         self.weight = Parameter(torch.FloatTensor(out_features, in_features))
#         nn.init.xavier_uniform_(self.weight)
        
#         self.s = s
#         self.m = m
#         self.cos_m = math.cos(m)
#         self.sin_m = math.sin(m)
    
#     def _calculate_para(self, s, m):
#         self.s = s
#         self.m = m
#         self.cos_m = math.cos(m)
#         self.sin_m = math.sin(m)

#     def forward(self, input, label, s, m):
#         self._calculate_para(s, m)
#         norm = torch.norm(input, dim=1, keepdim=True)
#         cosine = F.linear(F.normalize(input), F.normalize(self.weight))
#         sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
#         mod_cosine = torch.zeros_like(cosine)
#         mod_cosine = mod_cosine + cosine
#         for i in range(cosine.size(0)):
#             lb = int(label[i])
#             if cosine[i, lb] > 0:
#                 mod_cosine[i, lb] = cosine[i, lb] * self.cos_m - sine[i, lb] * self.sin_m
#             else:
#                 continue
#         return norm * mod_cosine

class AAMSoftmax_normfree(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=50.0, m=0.50):
        super(AAMSoftmax_normfree, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
    
    def _calculate_para(self, s, m):
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

    def forward(self, input, label, s, m):
        self._calculate_para(s, m)
        norm = torch.norm(input, dim=1, keepdim=True)
        nm_W = F.normalize(self.weight)
        cosine = F.linear(F.normalize(input), nm_W)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        mod_cosine = torch.zeros_like(cosine)
        mod_cosine = mod_cosine + cosine
        for i in range(cosine.size(0)):
            lb = int(label[i])
            if cosine[i, lb] > 0:
                mod_cosine[i, lb] = cosine[i, lb] * self.cos_m - sine[i, lb] * self.sin_m
            else:
                continue
        return norm * mod_cosine, nm_W

class AMSoftmax(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(AMSoftmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label, s, m):
        self.s = s
        self.m = m
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        margin = torch.zeros_like(cosine)

        for i in range(cosine.size(0)):
            lb = int(label[i])
            margin[i, lb] = self.m

        return self.s * (cosine - margin)

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
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
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

class ASoftmax(nn.Module):
    # inter speaker
    def __init__(self, in_features, out_features, m = 4, phiflag=True):
        super(ASoftmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

        cos_theta = x.mm(ww) # size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        cos_theta = cos_theta.clamp(-1,1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = Variable(cos_theta.data.acos())
            k = (self.m*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta,self.m)
            phi_theta = phi_theta.clamp(-1*self.m,1)

        cos_theta = cos_theta * xlen.view(-1,1)
        phi_theta = phi_theta * xlen.view(-1,1)
        output = (cos_theta,phi_theta)
        return output, ww # size=(B,Classnum,2)

# class ASoftmax(nn.Module):
#     # normal
#     def __init__(self, in_features, out_features, m = 4, phiflag=True):
#         super(ASoftmax, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = Parameter(torch.Tensor(in_features,out_features))
#         self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
#         self.phiflag = phiflag
#         self.m = m
#         self.mlambda = [
#             lambda x: x**0,
#             lambda x: x**1,
#             lambda x: 2*x**2-1,
#             lambda x: 4*x**3-3*x,
#             lambda x: 8*x**4-8*x**2+1,
#             lambda x: 16*x**5-20*x**3+5*x
#         ]

#     def forward(self, input):
#         x = input   # size=(B,F)    F is feature len
#         w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

#         ww = w.renorm(2,1,1e-5).mul(1e5)
#         xlen = x.pow(2).sum(1).pow(0.5) # size=B
#         wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

#         cos_theta = x.mm(ww) # size=(B,Classnum)
#         cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
#         cos_theta = cos_theta.clamp(-1,1)

#         if self.phiflag:
#             cos_m_theta = self.mlambda[self.m](cos_theta)
#             theta = Variable(cos_theta.data.acos())
#             k = (self.m*theta/3.14159265).floor()
#             n_one = k*0.0 - 1
#             phi_theta = (n_one**k) * cos_m_theta - 2*k
#         else:
#             theta = cos_theta.acos()
#             phi_theta = myphi(theta,self.m)
#             phi_theta = phi_theta.clamp(-1*self.m,1)

#         cos_theta = cos_theta * xlen.view(-1,1)
#         phi_theta = phi_theta * xlen.view(-1,1)
#         output = (cos_theta,phi_theta)
#         return output # size=(B,Classnum,2)

# class AngleLoss(nn.Module):
#     # fine tune
#     def __init__(self):
#         super(AngleLoss, self).__init__()
#         self.it = 0
#         self.LambdaMin = 5.0
#         self.LambdaMax = 1500.0 * 8
#         self.lamb = 1500.0 * 8

#     def forward(self, input, target):
#         if target.size()[0] != 1:
#             self.it += 1
#         cos_theta,phi_theta = input
#         target = target.view(-1,1) #size=(B,1)

#         index = cos_theta.data * 0.0 #size=(B,Classnum)
#         index.scatter_(1,target.data.view(-1,1),1)
#         index = index.byte()
#         index = Variable(index)

#         self.lamb = self.LambdaMin # fine tune

#         output = cos_theta * 1.0 #size=(B,Classnum)
#         output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
#         output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

#         logpt = F.log_softmax(output, dim=1)
#         logpt = logpt.gather(1,target)
#         logpt = logpt.view(-1)

#         loss = -1 * logpt
#         loss = loss.mean()

#         return loss

class AngleLoss(nn.Module):
    # resnet_mixup_a(should be the same as the orginal), use this to get normal trainig
    def __init__(self):
        super(AngleLoss, self).__init__()
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0 * 8
        self.lamb = 1500.0 * 8

    def forward(self, input, target):
        if target.size()[0] != 1:
            self.it += 1
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.byte()
        index = Variable(index)

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))

        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        logpt = F.log_softmax(output, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)

        loss = -1 * logpt
        loss = loss.mean()

        return loss

# class AngleLoss(nn.Module):
#     # resnet_a
#     def __init__(self, gamma=0):
#         super(AngleLoss, self).__init__()
#         self.gamma   = gamma
#         self.it = 0
#         self.LambdaMin = 5.0
#         self.LambdaMax = 1500.0 * 8
#         self.lamb = 1500.0 * 8

#     def forward(self, input, target):
#         if target.size()[0] != 1:
#             self.it += 1
#         cos_theta,phi_theta = input
#         target = target.view(-1,1) #size=(B,1)

#         index = cos_theta.data * 0.0 #size=(B,Classnum)
#         index.scatter_(1,target.data.view(-1,1),1)
#         index = index.byte()
#         index = Variable(index)

#         self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))

#         output = cos_theta * 1.0 #size=(B,Classnum)
#         output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
#         output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

#         logpt = F.log_softmax(output, dim=1)
#         logpt = logpt.gather(1,target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())

#         loss = -1 * (1-pt)**self.gamma * logpt
#         loss = loss.mean()

#         return loss

# class AngleLoss(nn.Module):
#     # resnet18_a_qc
#     def __init__(self, gamma=0):
#         super(AngleLoss, self).__init__()
#         self.gamma   = gamma
#         self.it = 0
#         self.LambdaMin = 5.0
#         self.LambdaMax = 1500.0 * 8
#         self.lamb = 1500.0 * 8

#     def forward(self, input, target):
#         if target.size()[0] != 1:
#             self.it += 1
#         cos_theta,phi_theta = input
#         target = target.view(-1,1) #size=(B,1)

#         index = cos_theta.data * 0.0 #size=(B,Classnum)
#         index.scatter_(1,target.data.view(-1,1),1)
#         index = index.byte()
#         index = Variable(index)

#         if self.it < 20000:
#             self.lamb = self.LambdaMax
#         else:
#             self.lamb = self.LambdaMin

#         output = cos_theta * 1.0 #size=(B,Classnum)
#         output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
#         output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

#         logpt = F.log_softmax(output, dim=1)
#         logpt = logpt.gather(1,target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())

#         loss = -1 * (1-pt)**self.gamma * logpt
#         loss = loss.mean()

#         return loss

# class AngleLoss(nn.Module):
#     # resnet_a_soft
#     def __init__(self, gamma=0):
#         super(AngleLoss, self).__init__()
#         self.gamma   = gamma
#         self.it = 0
#         self.LambdaMin = 5.0 * 3
#         self.LambdaMax = 1500.0 * 8
#         self.lamb = 1500.0 * 8

#     def forward(self, input, target):
#         if target.size()[0] != 1:
#             self.it += 1
#         cos_theta,phi_theta = input
#         target = target.view(-1,1) #size=(B,1)

#         index = cos_theta.data * 0.0 #size=(B,Classnum)
#         index.scatter_(1,target.data.view(-1,1),1)
#         index = index.byte()
#         index = Variable(index)

#         self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))

#         output = cos_theta * 1.0 #size=(B,Classnum)
#         output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
#         output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

#         logpt = F.log_softmax(output, dim=1)
#         logpt = logpt.gather(1,target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())

#         loss = -1 * (1-pt)**self.gamma * logpt
#         loss = loss.mean()

#         return loss
