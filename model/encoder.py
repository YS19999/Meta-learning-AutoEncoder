import torch
import torch.nn as nn
import torch.nn.functional as F

from embedding.rnn import RNN

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(input_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = self.tanh(x1)
        x2 = self.linear2(x1)
        x2 = x2.squeeze(dim=2)
        attn_weights = self.softmax(x2)
        attn_weights = attn_weights.unsqueeze(dim=2)
        weighted_input = torch.mul(attn_weights, x)
        output = weighted_input
        return output


class Encoder(nn.Module):
    def __init__(self, ebd, args):
        super(Encoder, self).__init__()

        self.ebd = ebd
        self.args = args

        self.ebd_embedding = self.ebd.embedding_dim

        self.ebd_dim = 64

        self.dropout = nn.Dropout(0.2)

        self.attn = Attention(self.ebd_dim)

        self.rnn = RNN(self.ebd_embedding, 32, 1, True, 0)

        self.encoder_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.ebd_dim, self.ebd_dim)
        )

        self.adv_head = nn.Linear(self.ebd_dim, 1)

    def forward(self, data, flag=None):

        ebd = self.ebd(data)

        d_ebd = self.dropout(ebd)

        rnn_ebd = self.rnn(d_ebd, data['text_len'])

        attn_ebd = self.attn(rnn_ebd)

        encoder = self.encoder_head(attn_ebd)

        seq = self.adv_head(attn_ebd).squeeze(-1)

        weight = F.softmax(seq, dim=-1)

        text_ebd = torch.sum(weight.unsqueeze(-1) * ebd, dim=1)

        if weight.shape[1] < 500:
            zero = torch.zeros((weight.shape[0], 500-weight.shape[1]))
            if self.args.cuda != -1:
               zero = zero.cuda(self.args.cuda)
            weight = torch.cat((weight, zero), dim=-1)
        else:
            weight = weight[:, :500]

        return text_ebd, weight, ebd, encoder
