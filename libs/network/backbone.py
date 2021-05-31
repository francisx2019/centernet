# -*- coding:utf-8 -*-
"""Feature Pyramid Network (FPN) on top of ResNet. Comes with task-specific
   heads on top of it.
See:
- https://arxiv.org/abs/1612.03144 - Feature Pyramid Networks for Object
  Detection
- http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf - A Unified
  Architecture for Instance and Semantic Segmentation
"""
import timm
import torch
import torch.nn as nn
from torchvision import models


# 设置relu函数的inplace=True
def convert_to_inplace_relu(model):
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = True


class ResNet(nn.Module):
    def __init__(self, num_layer='r50', pretrained=True):
        super(ResNet, self).__init__()
        if not pretrained:
            print("not loading pretrained weights.")
        if num_layer == 'r18':
            self.resnet = models.resnet18(pretrained=pretrained)
            num_bottleneck_filters = 512
        elif num_layer == 'r34':
            self.resnet = models.resnet34(pretrained=pretrained)
            num_bottleneck_filters = 512
        elif num_layer == 'r50':
            self.resnet = models.resnet50(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif num_layer == 'r101':
            self.resnet = models.resnet101(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif num_layer == 'r152':
            self.resnet = models.resnet50(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif num_layer == 'rx50':
            self.resnet = models.resnext50_32x4d(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif num_layer == 'rx101':
            self.resnet = models.resnext101_32x8d(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif num_layer == 'r50d':
            self.resnet = timm.create_model(model_name='gluon_resnet50_v1d',
                                            pretrained=pretrained)
            convert_to_inplace_relu(self.resnet)
            num_bottleneck_filters = 2048
        elif num_layer == 'r101d':
            self.resnet = timm.create_model(model_name='gluon_resnet101_v1d',
                                            pretrained=pretrained)
            convert_to_inplace_relu(self.resnet)
            num_bottleneck_filters = 2048
        else:
            assert False, "error num_layers."

        self.out_features = num_bottleneck_filters

    def forward(self, x):
        size = x.size()
        assert size[-1] % 32 == 0 and size[-2] % 32 == 0, "Image resolution has to be divisible by 32 for resnet"

        feat = self.resnet.conv1(x)
        feat = self.resnet.bn1(feat)
        feat = self.resnet.relu(feat)
        feat = self.resnet.maxpool(feat)

        x1 = self.resnet.layer1(feat)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        return x1, x2, x3, x4

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def freeze_stage(self, stage):
        if stage >= 0:
            self.resnet.bn1.eval()
            for m in [self.resnet.conv1, self.resnet.bn1]:
                for param in m.parameters():
                    param.requires_grad = False
        for i in range(1, stage + 1):
            layer = getattr(self.resnet, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False


if __name__ == '__main__':
    x = torch.randn((1, 3, 224, 224))
    model = ResNet(num_layer='rx50')
    output = model(x)
    for i in output:
        print(i.shape)
