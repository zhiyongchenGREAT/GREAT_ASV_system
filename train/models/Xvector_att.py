import torch
import torch.nn as nn
import torch.nn.functional as F

class self_attention_layer(nn.Module):
    def __init__(self, head_num=1, in_dim=1500):
        super(self_attention_layer, self).__init__()
        self.in_dim = in_dim
        self.head_num = head_num
        self.da = 512
        self.w_1 = nn.Parameter(torch.FloatTensor(self.in_dim, self.da))
        self.bias_1 = nn.Parameter(torch.FloatTensor(self.da))
        self.w_2 = nn.Parameter(torch.FloatTensor(self.da, self.head_num))
        self.bias_2 = nn.Parameter(torch.FloatTensor(self.head_num))
        self.relu = nn.ReLU(True)
        nn.init.xavier_uniform_(self.w_1)
        nn.init.zeros_(self.bias_1)
        nn.init.xavier_uniform_(self.w_2)
        nn.init.zeros_(self.bias_2)

    def forward(self,input_feature):
        # print(self.w_1[0, 50:60])
        # print(self.bias_1[50:60])
        batch_size = input_feature.size(0)
        feature_size = input_feature.size(1)
        A = F.softmax(torch.matmul(self.relu(torch.matmul(torch.transpose(input_feature, 1, 2),self.w_1)+self.bias_1),self.w_2)+self.bias_2,dim=1)
        E = torch.matmul(input_feature,A)
        std = torch.sqrt(self.relu(torch.matmul(input_feature*input_feature,A)-E*E)+0.00001)
        # std = torch.matmul(input_feature*input_feature,A)-E*E
        layer_out = torch.cat((E,std),1).reshape(batch_size,feature_size*self.head_num*2)
        # layer_out = E.reshape(batch_size,feature_size*self.head_num)
        
        return A, layer_out

# loss_array = torch.matmul(torch.transpose(a, 1, 2),a)-torch.eye(3)
# loss = torch.mean(torch.norm(loss_array, p='fro', dim=(1, 2), keepdim=True, out=None, dtype=None))