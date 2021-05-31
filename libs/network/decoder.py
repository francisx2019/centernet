# -*- coding:utf-8 -*-
"""
Author:xufei
Date:2021/4/28
"""
import torch
import torch.nn as nn


# get backbone feats: [bs, 2048, _h, _w]
class Decoder(nn.Module):
    def __init__(self, in_planes, bn_moment=0.1):
        super(Decoder, self).__init__()
        self.bn_moment = bn_moment
        self.in_planes = in_planes
        self.deconv_with_bias = False
        self.deconv_layers = self._make_deconv_layer(num_layers=3,
                                                     num_filters=[256, 256, 256],
                                                     num_kernels=[4, 4, 4])

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        for i in range(num_layers):
            kernel = num_kernels[i]
            padding = 0 if kernel == 2 else 1
            out_padding = 1 if kernel == 3 else 0
            planes = num_filters[i]
            layers.append(nn.ConvTranspose2d(in_channels=self.in_planes,
                                             out_channels=planes,
                                             kernel_size=kernel,
                                             stride=2,
                                             padding=padding,
                                             output_padding=out_padding,
                                             bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=self.bn_moment))
            layers.append(nn.ReLU(inplace=True))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv_layers(x)


if __name__ == '__main__':
    x = torch.randn((2, 2048, 56, 56))
    net = Decoder(in_planes=2048)
    out = net(x)
    print(out.size())
