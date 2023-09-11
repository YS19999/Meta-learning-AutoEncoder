import torch
import torch.nn as nn

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

class Decoder(nn.Module):
    def __init__(self, ebd_dim, ebd_embedding, args):
        super(Decoder, self).__init__()

        self.args = args
        self.ebd_dim = ebd_dim
        self.ebd_embedding = ebd_embedding

        self.attn = Attention(self.ebd_dim)

        self.d = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.ebd_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.ebd_embedding)
        )


    def forward(self, inputs):

        inputs = self.attn(inputs)

        output = self.d(inputs)

        return output

