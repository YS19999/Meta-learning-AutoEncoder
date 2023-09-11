import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.dis = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(500, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2),
        )

    def forward(self, reverse_feature):

        logits = torch.sigmoid(self.dis(reverse_feature))  # [b, 500] -> [b, 2]

        return logits
