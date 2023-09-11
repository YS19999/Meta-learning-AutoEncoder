import torch
import torch.nn as nn
import torch.nn.functional as F


class BASE(nn.Module):
    '''
        BASE model
    '''
    def __init__(self, args):
        super(BASE, self).__init__()
        self.args = args

        # cached tensor for speed
        self.I_way = nn.Parameter(torch.eye(self.args.way, dtype=torch.float), requires_grad=False)

    def _compute_l2(self, XS, XQ):

        diff = XS.unsqueeze(0) - XQ.unsqueeze(1)
        dist = torch.norm(diff, dim=2)

        return dist

    def _compute_cos(self, XS, XQ):

        dot = torch.matmul(
                XS.unsqueeze(0).unsqueeze(-2),
                XQ.unsqueeze(1).unsqueeze(-1)
                )
        dot = dot.squeeze(-1).squeeze(-1)

        scale = (torch.norm(XS, dim=1).unsqueeze(0) *
                 torch.norm(XQ, dim=1).unsqueeze(1))

        scale = torch.max(scale,
                          torch.ones_like(scale) * 1e-8)

        dist = 1 - dot/scale

        return dist

    def reidx_y(self, YS, YQ):

        unique1, inv_S = torch.unique(YS, sorted=True, return_inverse=True)
        unique2, inv_Q = torch.unique(YQ, sorted=True, return_inverse=True)

        if len(unique1) != len(unique2):
            raise ValueError(
                'Support set classes are different from the query set')

        if len(unique1) != self.args.way:
            raise ValueError(
                'Support set classes are different from the number of ways')

        if int(torch.sum(unique1 - unique2).item()) != 0:
            raise ValueError(
                'Support set classes are different from the query set classes')

        Y_new = torch.arange(start=0, end=self.args.way, dtype=unique1.dtype,
                device=unique1.device)

        return Y_new[inv_S], Y_new[inv_Q]

    def _init_mlp(self, in_d, hidden_ds, drop_rate):
        modules = []

        for d in hidden_ds[:-1]:
            modules.extend([
                nn.Dropout(drop_rate),
                nn.Linear(in_d, d),
                nn.ReLU()])
            in_d = d

        modules.extend([
            nn.Dropout(drop_rate),
            nn.Linear(in_d, hidden_ds[-1])])

        return nn.Sequential(*modules)

    def _label2onehot(self, Y):

        Y_onehot = F.embedding(Y, self.I_way)

        return Y_onehot

    @staticmethod
    def compute_acc(pred, true):

        return torch.mean((torch.argmax(pred, dim=1) == true).float()).item()
