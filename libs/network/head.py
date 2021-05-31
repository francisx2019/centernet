# -*- coding:utf-8 -*-
"""
Author:xufei
Date:2021/4/28
"""
import torch
import torch.nn as nn


class Head(nn.Module):
    def __init__(self, num_classes=20, channels=64):
        super(Head, self).__init__()
        # 热力图预测
        self.cls_head = nn.Sequential(nn.Conv2d(256, channels, kernel_size=3, padding=1, bias=False),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(channels, num_classes, kernel_size=1, stride=1, padding=0, bias=False))
        # 宽高预测
        self.wh_head = self.ConvReluConv(256, 2)
        # 中心点预测（x，y偏移情况）
        self.reg_head = self.ConvReluConv(256, 2)

    def ConvReluConv(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, bias=False),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False))

    def forward(self, x):
        cls = self.cls_head(x).sigmoid()
        wh = self.wh_head(x)
        offset = self.reg_head(x)
        return cls, wh, offset


if __name__ == '__main__':
    x = torch.randn((2, 256, 56, 56))
    net = Head()
    out = net(x)
    print(out[0].size(), out[1].size(), out[2].size())
