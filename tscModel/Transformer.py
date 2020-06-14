from torch import nn
import torch.nn.functional as F
import torch
import math
from torch.autograd import Variable
import matplotlib.pyplot as plt
import copy


class Transformer(nn.Module):

    def __init__(self, d_model, classes_nums, dropout, layer_nums, head_nums, device):
        super(Transformer, self).__init__()
        self.eb_layer = SequenceEncoding(d_model, dropout)
        self.pe_layer = PositionalEncoding(d_model, dropout, device)
        self.ed = Encoder(EncoderLayer(d_model,
                                       MultiHeadedAttention(head_nums, d_model, dropout),
                                       PositionwiseFeedForward(d_model, d_model*2)),
                          d_model,
                          layer_nums)
        self.lstm = nn.LSTM(d_model,d_model)
        self.output_layer = nn.Linear(d_model, classes_nums)
        self.DEVICE = device
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
        y = self.eb_layer(x)
        y = self.pe_layer(y)
        y = self.ed(y)
        y = self.lstm(y)[0][:, -1, :].squeeze()
        y = self.output_layer(y)
        return y


class Encoder(nn.Module):
    """
    编码器

    """

    def __init__(self, layer, d_model, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(d_model)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)



class EncoderLayer(nn.Module):
    """
    编码层

    """

    def __init__(self, d_model, self_attn, feed_forward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)

    def forward(self, x):
        x = self.sublayer[0](x, self.self_attn)
        x = self.sublayer[1](x, self.feed_forward)

        return x


class SublayerConnection(nn.Module):
    """
    残差连接

    """

    def __init__(self, d_model, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "add norm"
        return x + self.dropout(sublayer(self.norm(x))[0])


class LayerNorm(nn.Module):
    """
    LayerNorm

    """

    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        "Norm"
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # print(x.shape)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(nn.Module):
    """
    前馈神经网络

    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class MultiHeadedAttention(nn.Module):
    """
    多头注意力机制

    x:[B, L, D]
    query:[B, N, L, sub_dimension]
    key:[B, N, L, sub_dimension]
    value:[B, N, L, sub_dimension]

    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.h = h
        self.d_k = d_model // h
        self.q_layer = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1)
        self.k_layer = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1)
        self.v_layer = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        query, key, value = self.dropout(self.q_layer(x.transpose(1, 2)).transpose(1, 2)),\
                            self.dropout(self.k_layer(x.transpose(1, 2)).transpose(1, 2)),\
                            self.dropout(self.v_layer(x))
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

    def __init__(self, d_model, dropout, device, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)
        self.device = device
        # self.register_buffer('pe', self.pe)

    def forward(self, x):
        y = x + Variable(self.pe[:, : x.size(1), :], requires_grad=False).to(self.device)
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
# linears = clones(nn.Linear(512, 512), 4)
linears = nn.ModuleList([nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                         nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                         nn.Linear(512, 512)])
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
query=linears[0](y.transpose(1, 2)).view(nbatches, -1, h, d_k).transpose(1, 2)
key= linears[1](y.transpose(1, 2)).view(nbatches, -1, h, d_k).transpose(1, 2)
value = linears[2](y).view(nbatches, -1, h, d_k).transpose(1, 2)
print(query.shape, key.shape, value.shape)
result, atte = attention(query, key, value)
print(result.shape, atte.shape)
result = result.transpose(1, 2).contiguous().view(nbatches, -1, h * d_k)
print(result.shape)
'''



'''
eb_layer = SequenceEncoding(512, 0.1)
pe_layer = PositionalEncoding(512, 0.1)
attention = MultiHeadedAttention(8, 512, 0.1)
sc = clones(SublayerConnection(512), 2)
pf = PositionwiseFeedForward(512,1024)

x = torch.zeros(5, 1, 2000)
print(x.shape)
y = eb_layer(x)
print(y.shape)
y = pe_layer(y)
print(y.shape)
y = sc[0](y, attention)
print(y.shape)
y = sc[1](y, pf)
print(y.shape)'''
'''
eb_layer = SequenceEncoding(512, 0.1)
pe_layer = PositionalEncoding(512, 0.1)
attention = MultiHeadedAttention(8, 512, 0.1)
pff = PositionwiseFeedForward(512,1024)
el = EncoderLayer(512, attention, pff)
ed = Encoder(el, 512, 6)

x = torch.zeros(5, 1, 2000)
print(x.shape)
y = eb_layer(x)
print(y.shape)
y = pe_layer(y)
print(y.shape)
y = ed(y)
print(y.shape)
'''
'''device = 'cuda:0'
tf = Transformer(128, 10, 0.1, 6, 8, device).to(device)
x = torch.zeros(5, 2000)
print(x.shape)
y = tf(x)
print(y.shape)'''
