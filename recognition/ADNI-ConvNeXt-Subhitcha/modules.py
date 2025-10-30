# modules.py
import torch
import torch.nn as nn
from torchvision import ops

# ---------- Model blocks ----------
class LayerNorm2d(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias   = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.0, dropout: float = 0.2):
        super().__init__()
        self.dwconv   = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm     = LayerNorm2d(dim)
        self.pwconv1  = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act      = nn.GELU()
        self.dropout  = nn.Dropout(dropout)
        self.pwconv2  = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.gamma    = nn.Parameter(torch.ones(dim))
        self.droppath = drop_path

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.pwconv2(x)
        x = self.gamma[:, None, None] * x
        x = residual + ops.stochastic_depth(x, self.droppath, "row", self.training)
        return x

class ConvNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=2,
                 depths=[4,4,18,4], dims=[96,192,384,768],
                 drop_path_rate=0.1, dropout=0.2):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            ))

        self.stages = nn.ModuleList()
        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cur = 0
        for i in range(4):
            stage = nn.Sequential(*[
                ConvNeXtBlock(dims[i], dp_rates[cur + j], dropout) for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        return self.head(self.forward_features(x))

# ---------- Loss ----------
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=2, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing  = smoothing
        self.cls        = classes
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, target):
        logprobs = self.log_softmax(x)
        with torch.no_grad():
            true_dist = torch.zeros_like(logprobs)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * logprobs, dim=-1))

def build_model(num_classes=2, dropout=0.2, depths=[4,4,18,4]):
    return ConvNeXt(num_classes=num_classes, dropout=dropout, depths=depths)
