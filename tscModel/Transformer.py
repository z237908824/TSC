from torch import nn
import torch.nn.functional as F
import torch
import math
from torch.autograd import Variable


class Transformer(nn.Module):

    def __init__(self, ):
        super(Transformer, self).__init__()

    def forward(self, x):
        """
        输入: [batch_size, 1, seq_length]

        输出: [batch_size, num_classes]

        Transformer for TSC 模型
            编码层
                输入编码+位置编码
                multi-head self-attention 层
                position-wise feed-forward network
            特征提取层
            计划使用RNN结构或注意力机制

        """
        x = x.clone().detach().float().to(self.DEVICE).unsqueeze(1)
        # x.shape : [batch_size, 1, seq_length]
        return


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention mechanism.

    基本注意力机制单元

    """

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None):
        """
        前向传播

        Args:
        q: Queries张量，形状为[B, L_q, D_q]
        k: Keys张量，形状为[B, L_k, D_k]
        v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
        scale: 缩放因子，一个浮点标量

        Returns:
        上下文张量和attetention张量
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        # 计算softmax
        attention = self.softmax(attention)
        # 添加dropout
        attention = self.dropout(attention)
        # 和V做点积
        result = torch.bmm(attention, v)
        return result, attention

class PositionalEncoding(nn.Module):
    """
    位置编码
    """
    def __init__(self, d_model, dropout, max_len=5000 ):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arrange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)
        self.register_buffer('pe', self.pe)
    def forward(self, x):

        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)