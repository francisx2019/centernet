# -*- coding:utf-8 -*-
"""
Author:xufei
Date:2021/4/28
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from libs.network import FPN, Head, Decoder, ResNet


def gather_feature(feat, index, mask=None, use_transformer=False):
    if use_transformer:
        # (B, C, H, W) --> (B, C, HxW) --> (B, HxW, C)
        batch, channel = feat.shape[:2]
        feat = feat.view(batch, channel, -1).permute((0, 2, 1)).contiguous()

    dim = feat.size(-1)
    index = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
    feat = feat.gather(dim=1, index=index)

    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.reshape(-1, dim)
    return feat


class CenterNet(nn.Module):
    def __init__(self, cfg):
        super(CenterNet, self).__init__()
        self._fpn = cfg.MODEL.FPN
        self.down_stride = cfg.MODEL.down_stride
        self.score_theta = cfg.MODEL.score_theta
        self.classes_name = cfg.DATA.classes_name
        self.num_classes = cfg.MODEL.num_classes
        self.backbone = ResNet(cfg.MODEL.num_layer)
        if self._fpn:
            self.fpn = FPN(self.backbone.out_features)
        self.upsample = Decoder(self.backbone.out_features if not cfg.MODEL.FPN else 2048, cfg.MODEL.bn_momentum)
        self.head = Head(channels=cfg.MODEL.head_channel, num_classes=self.num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        if self._fpn:
            feat = self.fpn(feats)
        else:
            feat = feats[-1]
        return self.head(self.upsample(feat))

    @torch.no_grad()
    def inference(self, img, topk=40, return_hm=False, theta=None):
        feats = self.backbone(img)
        if self._fpn:
            feat = self.fpn(feats)
        else:
            feat = feats[-1]
        # (b, cls, h, w) -- (b, points, h, w) -- (b, points, h, w)
        pred_hm, pred_wh, pred_offset = self.head(self.upsample(feat))

        _, _, h, w = img.shape
        b, c, out_h, out_w = pred_hm.shape
        # 经过此函数后，把每个像素的最大值获取，其他置为0
        pred_hm = self.pool_nms(pred_hm)
        scores, index, clses, ys, xs = self.topk_score(pred_hm, k=topk)
        reg = gather_feature(pred_offset, index, use_transformer=True)
        reg = reg.reshape(b, topk, 2)
        xs = xs.view(b, topk, 1) + reg[:, :, 0:1]
        ys = ys.view(b, topk, 1) + reg[:, :, 1:2]

        clses = clses.reshape(b, topk, 1).float()
        scores = scores.reshape(b, topk, 1)

        wh = gather_feature(pred_wh, index, use_transformer=True)
        wh = wh.reshape(b, topk, 2)
        half_w, half_h = wh[..., 0:1] / 2, wh[..., 1:2] / 2
        bboxes = torch.cat([xs-half_w, ys-half_h, xs+half_w, ys+half_h], dim=2)

        detects = []
        for batch in range(b):
            mask = scores[batch].gt(self.score_theta if theta is None else theta)   # 获取大于score阈值的像素
            batch_boxes = bboxes[batch][mask.squeeze(-1), :]
            batch_boxes[:, [0, 2]] *= w / out_w
            batch_boxes[:, [1, 3]] *= h / out_h
            batch_scores = scores[batch][mask]
            batch_cls = clses[batch][mask]
            batch_cls = [self.classes_name[int(cls.item())] for cls in batch_cls]
            detects.append([batch_boxes, batch_scores, batch_cls, pred_hm[batch] if return_hm else None])
        return detects

    @staticmethod
    def pool_nms(pred, pool_size=3):
        pad = (pool_size - 1) // 2
        hm_max = F.max_pool2d(pred, pool_size, stride=1, padding=pad)
        keep = (hm_max == pred).float()
        return pred * keep

    @staticmethod
    def topk_score(scores, k):
        batch, channel, height, width = scores.shape
        # 找出所有像素中前k个像素可能包含目标物体
        topk_scores, topk_inds = torch.topk(scores.reshape(batch, channel, -1), k)
        topk_inds = topk_inds % (height * width)

        # 获取位置相对与图片长宽的坐标
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, index = torch.topk(topk_scores.reshape(batch, -1), k)
        topk_cls = (index / k).int()
        topk_inds = gather_feature(topk_inds.view(batch, -1, 1), index).reshape(batch, k)
        topk_ys = gather_feature(topk_ys.view(batch, -1, 1), index).reshape(batch, k)
        topk_xs = gather_feature(topk_xs.view(batch, -1, 1), index).reshape(batch, k)
        return topk_score, topk_inds, topk_cls, topk_ys, topk_xs


if __name__ == '__main__':
    # from utils.config import config
    # cfg = config()
    # model = CenterNet(cfg=cfg)
    # x = torch.randn((1, 3, 512, 512))
    # out = model(x)
    # print(out[0].size(), out[1].size(), out[2].size())
    x = torch.Tensor([[1,2,3], [4,5,6]])
    indx = torch.LongTensor([[0,1,0], [1,0,1]])
    print(x.gather(dim=0, index=indx))