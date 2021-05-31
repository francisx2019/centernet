# -*- coding:utf-8 -*-
"""
Author:xufei
Date:2021/4/28
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
bn = nn.BatchNorm2d


class ConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=1):
        super(ConvBNRelu, self).__init__()
        if kernel_size == 3:
            self.block = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 padding=padding,
                                                 bias=False),
                                       bn(out_channels),
                                       nn.ReLU(inplace=True))
        else:
            self.block = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 bias=False),
                                       bn(out_channels),
                                       nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels=512):
        super(FPN, self).__init__()
        self.laterals = nn.Sequential(*[ConvBNRelu(in_channels//(2**c), out_channels) for c in reversed(range(4))])
        self.smooth = nn.Sequential(*[ConvBNRelu(out_channels*c, out_channels*c, kernel_size=3) for c in range(1, 5)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, feats):
        laterals = [layer(feats[i]) for i, layer in enumerate(self.laterals)]

        out4 = laterals[-1]
        out3 = laterals[-2] + F.interpolate(out4, scale_factor=2, mode='nearest')
        out2 = laterals[-3] + F.interpolate(out3, scale_factor=2, mode='nearest')
        out1 = laterals[-4] + F.interpolate(out2, scale_factor=2, mode='nearest')

        out1 = self.smooth[0](out1)
        out2 = self.smooth[1](torch.cat([out2, self.pool(out1)], dim=1))
        out3 = self.smooth[2](torch.cat([out3, self.pool(out2)], dim=1))
        out4 = self.smooth[3](torch.cat([out4, self.pool(out3)], dim=1))
        return out4


if __name__ == '__main__':
    feats = [torch.randn((2, 256*2**(4-c), 7*2**(c-1), 7*2**(c-1))) for c in range(1, 5)][::-1]
    net = FPN(2048)
    out = net(feats)
    print(out.size())
