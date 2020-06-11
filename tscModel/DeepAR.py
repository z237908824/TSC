from torch import nn
import torch.nn.functional as F
import torch


class DeepAR(nn.Module):

    def __init__(self, hidden_size, classnum, device, num_layers=1, batch_first=True, dropout=0, bidirectional=True):
        super(DeepAR, self).__init__()
        self.DEVICE = device
        self.embedding_layer = embedding_layer(hidden_size)
        self.lstm = nn.LSTM(hidden_size,
                            hidden_size,
                            num_layers,
                            batch_first=batch_first,
                            dropout=dropout,
                            bidirectional=bidirectional)
        self.attention_layer = attention(hidden_size, int(bidirectional) + 1)
        self.output_layer = nn.Linear(hidden_size, classnum)

    def forward(self, x):
        x = x.clone().detach().float().to(self.DEVICE)
        x = self.embedding_layer(x)
        output, _ = self.lstm(x)
        output, alpha = self.attention_layer(output)

        out = self.output_layer(output)
        out = F.softmax(out, dim=1)

        return out


class attention(nn.Module):

    def __init__(self, hidden_size, bidirectional):
        super(attention, self).__init__()

        # 此处可以并行化优化
        self.k_net = nn.Linear(hidden_size * bidirectional, hidden_size)
        self.q_net = nn.Linear(hidden_size * bidirectional, hidden_size)
        self.v_net = nn.Linear(hidden_size * bidirectional, hidden_size)

    def forward(self, r_out):
        # r_out shape (batch, time_step, output_size*bi_lstm)
        key = self.k_net(r_out)
        query = self.q_net(r_out)
        value = self.v_net(r_out)
        alpha = key.mul(query)
        alpha = torch.nn.functional.adaptive_avg_pool1d(alpha, 1)
        alpha = F.softmax(alpha, dim=1)
        out = torch.sum(value * alpha, 1)
        return out, alpha


class embedding_layer(nn.Module):

    def __init__(self, hidden_size):
        super(embedding_layer, self).__init__()

        self.network = nn.Linear(1, hidden_size)

    def forward(self, x):
        # rcs shape (batch, time_step)
        x = torch.unsqueeze(x, dim=2)
        out = self.network(x)
        # rcs shape (batch, time_step,512)
        return out