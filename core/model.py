import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

channels = 256
num_res_block = 19
num_conv_block = 1

class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.action_size = 4864
        self.conv1 = nn.Conv2d(119, channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

    def forward(self, s):
        s = F.relu(self.bn1(self.conv1(s)))
        return s

class ResNetBlock(nn.Module):
    def __init__(self, inplanes = channels, planes = channels, stride = 1, downsample = None):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                    padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out

class OutBlock(nn.Module):  # Đầu ra value và policy
    def __init__(self):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(channels,out_channels= 1, kernel_size= 1)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(8 * 8, 64)
        self.fc2 = nn.Linear(64, 1)

        self.conv1 = nn.Conv2d(channels, 128, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(8 * 8 * 128, 4864)

    def forward(self, s, mask):
        # Value
        v = F.relu(self.bn(self.conv(s)))
        v = v.flatten(start_dim=1)  # tương đương với v.view(-1, 8*8)
        v = F.relu(self.fc1(v))
        v = F.tanh(self.fc2(v))

        # Policy
        p = F.relu(self.bn1(self.conv1(s)))
        p = p.view(-1, 8 * 8 * 128)
        p = self.fc(p)

        # Apply mask and normalize
        # p = p * mask  # Zero out illegal moves
        # p = F.softmax(p, dim=-1)  # Convert to probabilities
        # p = p * mask  # Re-mask to ensure only legal moves have non-zero probabilities
        # p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)  # Renormalize
        p = p.masked_fill(mask == 0, -1e9)  # mask out illegal moves
        p = F.softmax(p, dim=-1)
        
        return p, v

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()

        self.conv_blocks = nn.ModuleList([ConvBlock() for _ in range(num_conv_block)])

        self.res_blocks = nn.ModuleList([ResNetBlock() for _ in range(num_res_block)])

        self.outblock = OutBlock()

    def forward(self, s, mask):
        for conv in self.conv_blocks:
            s = conv(s)

        for res in self.res_blocks:
            s = res(s)

        p, v = self.outblock(s, mask)
        return p, v

class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, y_value, value, y_policy, policy):
        value_error = (value - y_value) ** 2
        policy_error = torch.sum((-policy*
                                (1e-6 + y_policy.float()).float().log()), 1)
        total_error = (value_error.view(-1).float() + policy_error).mean()
        return total_error