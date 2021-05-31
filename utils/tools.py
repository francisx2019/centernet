# -*- coding:utf-8 -*-
"""
Author:xufei
Date:2021/5/6
"""
import torch
import torch.nn.functional as F


def map2coord(h, w, stride):
    shift_x = torch.arange(0, w*stride, step=stride, dtype=torch.float32)
    shift_y = torch.arange(0, h*stride, step=stride, dtype=torch.float32)
    shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
    shift_x, shift_y = shift_x.reshape(-1), shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations


def gather_feats(featmap, index, mask=None, use_transform=False):
    if use_transform:
        # [N, C, H, W] --> [N, HxW, C]
        batch, channel = featmap.shape[:2]
        featmap = featmap.view(batch, channel, -1).permute((0, 2, 1)).contiguous()

    dim = featmap.size(-1)
    index = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
    featmap = featmap.gather(dim=1, index=index)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(featmap)
        featmap = featmap[mask]
        featmap = featmap.reshape(-1, dim)
    return featmap


if __name__ == '__main__':
    pass
