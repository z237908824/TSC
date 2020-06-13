from torch import nn
import torch.nn.functional as F
import torch
import math
from torch.autograd import Variable
import matplotlib.pyplot as plt
import copy


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


class Encoder(nn.Module):
    """
    编码器

    """

    def __init__(self, layer, N):
        super(Encoder, self).__init__()

    def forward(self, x, mask):
        return


class EncoderLayer(nn.Module):
    """
    编码层

    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()

    def forward(self, x, mask):
        return


class LayerNorm(nn.Module):
    """
    LayerNorm

    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()

    def forward(self, x):
        return


class SublayerConnection(nn.Module):
    """
    残差连接

    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()

    def forward(self, x, sublayer):
        return


class PositionwiseFeedForward(nn.Module):
    """
    前馈神经网络

    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

    def forward(self, x):
        return


class MultiHeadedAttention(nn.Module):
    """
    多头注意力机制

    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.h = h
        self.d_k = d_model // h
        self.q_layer = nn.Linear(d_model, d_model)
        self.k_layer = nn.Linear(d_model, d_model)
        self.v_layer = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        query, key, value = self.dropout(self.q_layer(x)), self.dropout(self.k_layer(x)), self.dropout(self.v_layer(x))
        result, atte = self.attention(query, key, value)
        result = result.transpose(1, 2).contiguous().view(x.size(0), -1, self.h * self.d_k)
        return result, atte


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
        上下文张量和attention张量
        """
        attention = torch.matmul(q, k.transpose(-2, -1))
        scale = math.sqrt(q.size(-1))
        if scale:
            attention = attention / scale
        # 计算softmax
        attention = self.softmax(attention)
        # 添加dropout
        attention = self.dropout(attention)
        # 和V做点积
        result = torch.matmul(attention, v)
        return result, attention


class PositionalEncoding(nn.Module):
    """
    位置编码

    用来记录序列中元素的位置信息
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)
        # self.register_buffer('pe', self.pe)

    def forward(self, x):
        y = x + Variable(self.pe[:, : x.size(1), :], requires_grad=False)
        return self.dropout(y)


class SequenceEncoding(nn.Module):
    """
    序列编码

    将形如[batch_size, 1, seq_length]的输入转化为[batch_size, d_model, seq_length]
    从而增加通道数

    考虑到单变量序列的特殊性，建议编码层添加bias
    """

    def __init__(self, d_model, dropout):
        super(SequenceEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.eb_layer = nn.Linear(1, d_model, bias=True)

    def forward(self, x):
        x = self.eb_layer(x.transpose(1, 2))
        return self.dropout(x)


def clones(module, N):
    "产生N个相同的层"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
'''
eb_layer = SequenceEncoding(512, 0.1)
pe_layer = PositionalEncoding(512, 0.1)
x = torch.zeros(5, 1, 2000)
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(224)
print(x.shape)
ax1.plot(x[0].squeeze().detach().numpy())
y = eb_layer(x)
ax2.imshow(y[0].detach().numpy())
print(y.shape)
y = pe_layer(y)
ax3.imshow(y[0].detach().numpy())
print(y.shape)
plt.show()
'''


'''
linears = clones(nn.Linear(512, 512), 4)
h = 8
d_k = 512 // 8
eb_layer = SequenceEncoding(512, 0.1)
pe_layer = PositionalEncoding(512, 0.1)
x = torch.zeros(5, 1, 2000)
attention = ScaledDotProductAttention(0.1)
print(x.shape)
y = eb_layer(x)
print(y.shape)
y = pe_layer(y)
print(y.shape)
nbatches = y.size(0)
query, key, value = [l(x).view(nbatches, -1, h, d_k).transpose(1, 2) for l, x in zip(linears, (y, y, y))]
print(query.shape, key.shape, value.shape)
result, atte = attention(query, key, value)
print(result.shape, atte.shape)
result = result.transpose(1, 2).contiguous().view(nbatches, -1, h * d_k)
print(result.shape)
'''
eb_layer = SequenceEncoding(512, 0.1)
pe_layer = PositionalEncoding(512, 0.1)
attention = MultiHeadedAttention(8, 512, 0.1)
x = torch.zeros(5, 1, 2000)
print(x.shape)
y = eb_layer(x)
print(y.shape)
y = pe_layer(y)
print(y.shape)
y, atte = attention(y)
print(y.shape)