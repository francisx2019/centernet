# -*- coding:utf-8 -*-
"""
Author:xufei
Date:2021/5/6
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.tools import gather_feats, map2coord


class Loss(nn.Module):
    def __init__(self, cfg):
        super(Loss, self).__init__()
        self.down_stride = cfg.MODEL.down_stride
        self.focal_loss = modified_focal_loss
        self.iou_loss = DIOULoss
        self.l1_loss = F.l1_loss

        self.alpha = cfg.LOSS.loss_alpha
        self.beta = cfg.LOSS.loss_beta
        self.gamma = cfg.LOSS.loss_gamma

    def forward(self, pre, gt):
        pre_heatmap, pre_wh, pre_offset = pre
        imgs, gt_boxes, gt_classes, gt_hm, infos = gt
        gt_nonpad_mask = gt_classes.gt(0)
        # print('pred_hm: ', pred_hm.shape, '  gt_hm: ', gt_hm.shape)
        cls_loss = self.focal_loss(pre_heatmap, gt_hm)

        # ------------------------ focal-loss -------------------------- #
        wh_loss = cls_loss.new_tensor(0.)
        offset_loss = cls_loss.new_tensor(0.)
        num = 0
        for batch in range(imgs.size(0)):
            ct = infos[batch]['center'].cuda()
            ct_int = ct.long()
            num += len(ct_int)
            batch_pos_pred_wh = pre_wh[batch, :, ct_int[:, 1], ct_int[:, 0]].view(-1)
            batch_pos_pred_offset = pre_offset[batch, :, ct_int[:, 1], ct_int[:, 0]].view(-1)

            batch_boxes = gt_boxes[batch][gt_nonpad_mask[batch]]
            wh = torch.stack([batch_boxes[:, 2] - batch_boxes[:, 0],
                              batch_boxes[:, 3] - batch_boxes[:, 1]]).view(-1) / self.down_stride

            offset = (ct - ct_int.float()).T.contiguous().view(-1)

            wh_loss += self.l1_loss(batch_pos_pred_wh, wh, reduction='sum')
            offset_loss += self.l1_loss(batch_pos_pred_offset, offset, reduction='sum')

        regr_loss = wh_loss * self.beta + offset_loss * self.gamma
        return cls_loss * self.alpha, regr_loss / (num + 1e-6)

        # ----------------------- IOU LOSS --------------------------#
        # output_h, output_w = pre_heatmap.shape[-2:]
        # b, _, h, w = imgs.shape
        # location = map2coord(output_h, output_w, self.down_stride).cuda()
        #
        # location = location.view(output_h, output_w, 2)
        # pre_offset = pre_offset.permute(0, 2, 3, 1)
        # pre_wh = pre_wh.permute(0, 2, 3, 1)
        # iou_loss = cls_loss.new_tensor(0.)
        # for batch in range(b):
        #     ct = infos[batch]['ct']
        #     xs, ys, pos_w, pos_h, pos_offset_x, pos_offset_y = [[] for _ in range(6)]
        #     for i, cls in enumerate(gt_classes[batch][gt_nonpad_mask[batch]]):
        #         ct_int = ct[i]
        #         xs.append(location[ct_int[1], ct_int[0], 0])
        #         ys.append(location[ct_int[1], ct_int[0], 1])
        #         pos_w.append(pre_wh[batch, ct_int[1], ct_int[0], 0])
        #         pos_h.append(pre_wh[batch, ct_int[1], ct_int[0], 1])
        #         pos_offset_x.append(pre_offset[batch, ct_int[1], ct_int[0], 0])
        #         pos_offset_y.append(pre_offset[batch, ct_int[1], ct_int[0], 1])
        #     xs, ys, pos_w, pos_h, pos_offset_x, pos_offset_y = \
        #         [torch.stack(i) for i in [xs, ys, pos_w, pos_h, pos_offset_x, pos_offset_y]]
        #
        #     det_boxes = torch.stack([xs - pos_w / 2 + pos_offset_x, ys - pos_h / 2 + pos_offset_y,
        #                              xs + pos_w / 2 + pos_offset_x, ys + pos_h / 2 + pos_offset_y]).T.round()
        #
        #     iou_loss += self.iou_loss(det_boxes, gt_boxes[batch][gt_nonpad_mask[batch]])
        #
        # return cls_loss * self.alpha,  iou_loss / b * self.beta


# 回归框loss
def DIOULoss(pred, gt, size_num=True):
    if size_num:
        return torch.sum(1. - bbox_overlaps_diou(pred, gt)) / pred.size(0)
    return torch.sum(1. - bbox_overlaps_diou(pred, gt))


# 回归 l1_loss
def reg_l1_loss(output, mask, index, target):
    pred = gather_feats(output, index, use_transform=True)
    mask = mask.unsqueeze(dim=2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss


def bbox_overlaps_diou(bboxes1, bboxes2):
    rows, cols = bboxes1.shape[0], bboxes2.shape[0]
    dious = torch.zeros((rows, cols))
    if rows*cols == 0:
        return dious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        dious = torch.zeros((cols, rows))
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2
    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1 + area2 - inter_area
    dious = inter_area / union - (inter_diag) / outer_diag
    dious = torch.clamp(dious, min=-1.0, max=1.0)
    if exchange:
        dious = dious.T
    return dious


def modified_focal_loss(pred, gt):
    '''
    Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch, c, h, w)
        gt (batch, c, h, w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)
    # clamp min value is set to 1e-12 to maintain the numerical stability
    pred = torch.clamp(pred, 1e-12)
    # print(pred.size(), gt.size())

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


if __name__ == '__main__':
    pass
