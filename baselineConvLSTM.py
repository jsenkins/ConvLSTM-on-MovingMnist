
import torch.nn as nn
import torch.nn.functional as F

import torch


class ConvLSTMCell(nn.Module):
    def __init__(self, in_dims, hidden_dims, kernel_size=3):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dims = hidden_dims
        padding = kernel_size // 2

        # creating one conv for all matrix generation
        self.conv = nn.Conv2d(
            in_dims + hidden_dims, hidden_dims * 4, kernel_size, padding=padding
        )  # includes bias

    def forward(self, x_t, prev):
        h_t, c_t = prev

        # concatenating input and hidden by channel dimension
        x = torch.cat([x_t, h_t], dim=1)
        i, f, o, g = torch.split(self.conv(x), self.hidden_dims, dim=1)
        # forget gate
        
        c = torch.sigmoid(f) * c_t + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)

        return (h, c)

class ConvLSTMForecaster(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.hidden=hidden
        self.cell1 = ConvLSTMCell(16, hidden)      
        
        self.cell2 = ConvLSTMCell(hidden, hidden)      
        self.head = nn.Conv2d(64, 16, kernel_size=1)

    def forward(self, X):                      
        B = X.shape[0]
        states=None
        outputs, pred = [], None
        for tau in range(9):                           
            frame = X[:, tau] if tau < 5 else pred     
            u = F.pixel_unshuffle(frame, 4)
            if states is None:
                z = lambda: u.new_zeros(B, self.hidden, *u.shape[-2:])
                states = [(z(), z()), (z(), z())]
            (h1, c1), (h2, c2) = states
            h1, c1 = self.cell1(u, (h1, c1))
            h2, c2 = self.cell2(h1, (h2, c2))  
            states = [(h1, c1), (h2, c2)]
            pred = torch.sigmoid(F.pixel_shuffle(self.head(h2), 4))
            if tau >= 4:                               
                outputs.append(pred)
        return torch.stack(outputs, dim=1)

torch.manual_seed(0)
m = ConvLSTMForecaster()
n = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(n); assert n == 480_784
print(m(torch.rand(2, 5, 1, 64, 64)).shape) 
