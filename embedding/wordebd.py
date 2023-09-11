import torch.nn as nn
import torch.nn.functional as F


class WORDEBD(nn.Module):

    def __init__(self, vocab, finetune_ebd):
        super(WORDEBD, self).__init__()

        self.vocab_size, self.embedding_dim = vocab.vectors.size()
        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding_layer.weight.data = vocab.vectors

        self.finetune_ebd = finetune_ebd

        if self.finetune_ebd:
            self.embedding_layer.weight.requires_grad = True
        else:
            self.embedding_layer.weight.requires_grad = False

    def forward(self, data, weights=None):

        if (weights is None) or (self.finetune_ebd == False):
            ebd = self.embedding_layer(data['text'])
        else:
            ebd = F.embedding(data['text'], weights['ebd.embedding_layer.weight'])

        return ebd
