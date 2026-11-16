import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
from einops import rearrange


class MultiHeadExternalAttention(nn.Module):
    def __init__(self, feature_dim, external_dim, num_heads=8):
        """
        :param feature_dim: 输入特征的维度。
        :param external_dim: 外部空间的维度。
        :param num_heads: 注意力头的数量。
        """
        super(MultiHeadExternalAttention, self).__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.head_dim = external_dim // num_heads
        
        # 线性层用于生成Q, K, V
        # 有些东西，难以用线性组合表示
        self.fc1 = nn.Linear(feature_dim, feature_dim*2)
        # self.fc2 = nn.Linear(feature_dim, feature_dim)
        self.fc3 = nn.Linear(feature_dim, feature_dim) # 有些东西啊，就是这么地玄学，解释不通的
        self.fc4 = nn.Linear(feature_dim*2, feature_dim*2)


        assert external_dim % num_heads == 0, "external_dim must be divisible by num_heads"

    def forward(self, featuresI, featuresT, k=None): #(bs, dim)
        bs, dim = featuresI.shape
        VI = self.fc3(featuresI)
        VT = self.fc3(featuresT)
        # outI = torch.sum(VI,dim=0)*0.0078
        # outI = outI.unsqueeze(0).repeat(bs,1)
        # outT = torch.sum(VT,dim=0)*0.0078
        # outT = outT.unsqueeze(0).repeat(bs,1)
        # W = torch.ones(bs,bs).to(featuresI.device).half()*0.0078
        W = torch.ones(bs,bs).to(featuresI.device).half()*(1/bs)
        outI = torch.einsum("qk,kd->qd",[W,VI])
        outT = torch.einsum("qk,kd->qd",[W,VT])

        return outI, outT

