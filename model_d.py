import torch
from torch import nn
import torch.nn.functional as F


class ConvLSTMCellD(nn.Module):
    def __init__(self, in_ch, hid, k=3, dilation=1):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.conv = nn.Conv2d(
            in_ch + hid, 4 * hid, k, stride=1, padding=pad,
            dilation=dilation, bias=True
        )

    def forward(self, u, state):
        h, c = state
        i, f, o, g = self.conv(torch.cat([u, h], dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTMForecasterD(nn.Module):
    def __init__(self, patch=4, kernels=(3, 3), dilations=(1, 1), hidden=64):
        super().__init__()
        self.patch, self.hidden = patch, hidden
        self.cell1 = ConvLSTMCellD(patch * patch, hidden, kernels[0], dilations[0])
        self.cell2 = ConvLSTMCellD(hidden, hidden, kernels[1], dilations[1])
        self.head = nn.Conv2d(hidden, patch * patch, 1)

    def forward(self, X):
        B, states, outputs, pred = X.shape[0], None, [], None
        for tau in range(9):
            frame = X[:, tau] if tau < 5 else pred
            u = F.pixel_unshuffle(frame, self.patch)
            if states is None:
                z = lambda: u.new_zeros(B, self.hidden, *u.shape[-2:])
                states = [(z(), z()), (z(), z())]
            (h1, c1), (h2, c2) = states
            h1, c1 = self.cell1(u, (h1, c1))
            h2, c2 = self.cell2(h1, (h2, c2))
            states = [(h1, c1), (h2, c2)]
            pred = torch.sigmoid(F.pixel_shuffle(self.head(h2), self.patch))
            if tau >= 4:
                outputs.append(pred)
        return torch.stack(outputs, dim=1)


def reach_px(patch=4, kernels=(3, 3), dilations=(1, 1)):
    r1, r2 = (d * (k - 1) // 2 for k, d in zip(kernels, dilations))
    return patch * (r1 + r2)
