import torch
import torch.nn as nn
import torch.nn.functional as F

from classifier.base import BASE

class R2D2(BASE):

    def __init__(self, ebd_dim, args):
        super(R2D2, self).__init__(args)
        self.ebd_dim = ebd_dim

        self.args = args

        # meta parameters to learn
        self.lam = nn.Parameter(torch.tensor(-1, dtype=torch.float))
        self.alpha = nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.beta = nn.Parameter(torch.tensor(1, dtype=torch.float))
        # lambda and alpha is learned in the log space

        # cached tensor for speed
        self.I_support = nn.Parameter(torch.eye(self.args.shot * self.args.way, dtype=torch.float), requires_grad=False)
        self.I_way = nn.Parameter(torch.eye(self.args.way, dtype=torch.float), requires_grad=False)

    def _compute_w(self, XS, YS_onehot):

        W = XS.t() @ torch.inverse(XS @ XS.t() + (10. ** self.lam) * self.I_support) @ YS_onehot

        return W

    def _label2onehot(self, Y):

        Y_onehot = F.embedding(Y, self.I_way)

        return Y_onehot

    def forward(self, XS, YS, XQ, YQ, XQ_logitsD, XSource_logitsD, YQ_d, YSource_d):

        YS, YQ = self.reidx_y(YS, YQ)

        YS_onehot = self._label2onehot(YS)

        W = self._compute_w(XS, YS_onehot)

        pred = (10.0 ** self.alpha) * XQ @ W + self.beta

        loss = F.cross_entropy(pred, YQ)

        acc = BASE.compute_acc(pred, YQ)

        d_acc = (BASE.compute_acc(XQ_logitsD, YQ_d) + BASE.compute_acc(XSource_logitsD, YSource_d)) / 2

        return acc, d_acc, loss