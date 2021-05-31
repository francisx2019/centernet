# -*- coding:utf-8 -*-
"""
Author:xufei
Date:2021/5/6
"""
import torch
import numpy as np
import albumentations as A
import torch.nn.functional as F


# 翻转
def flip(img):
    return img[:, :, ::-1].copy()


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]             # 返回shape为[n,1]和[1,m]
    h = np.exp(-(x*x+y*y) / (2*sigma*sigma))    # h=[m,n]
    h[h < np.finfo(h.dtype).eps*h.max()] = 0    #
    return h


# 获取高斯模糊半径
def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    a1 = 1
    b1 = (height + width)
    c1 = width*height*(1-min_overlap)/(1+min_overlap)
    sq1 = np.sqrt(b1**2 - 4*a1*c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2*(height+width)
    c2 = (1-min_overlap)*width*height
    sq2 = np.sqrt(b2**2-4*a2*c2)
    r2 = (b2 + sq2) / 2

    a3 = 4*min_overlap
    b3 = -2*min_overlap*(height+width)
    c3 = (min_overlap-1)*width*height
    sq3 = np.sqrt(b3**2-4*a3*c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2*radius+1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter/6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width-x, radius+1)
    top, bottom = min(y, radius), min(height-y, radius+1)

    mask_heatmap = heatmap[y-top:y+bottom, x-left:x+right]
    masked_gaussian = gaussian[radius-top:radius+bottom, radius-left:radius+right]
    if min(masked_gaussian.shape) > 0 and min(mask_heatmap.shape) > 0:
        np.maximum(mask_heatmap, masked_gaussian*k, out=mask_heatmap)
    return heatmap


def draw_msra_gaussian(heatmap, center, sigma):
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return heatmap


def collate_fn(data):
    imgs_list, boxes_list, class_list, heatmap_list, infos = zip(*data)
    assert len(imgs_list) == len(boxes_list) == len(class_list), \
        f'img sample size:{len(imgs_list)}, boxes sample size:{len(boxes_list)}, class sample size:{len(class_list)}'

    batch_size = len(boxes_list)
    pad_imgs_list = []
    pad_boxes_list = []
    pad_class_list = []
    pad_heatmap_list = []
    h_list = [int(s.shape[1]) for s in imgs_list]
    w_list = [int(s.shape[2]) for s in imgs_list]
    max_h = np.array(h_list).max()
    max_w = np.array(w_list).max()

    for i in range(batch_size):
        img = imgs_list[i]
        heatmap = heatmap_list[i]

        pad_imgs_list.append(F.pad(img,
                                   (0, int(max_w - img.shape[2]), 0, int(max_h - img.shape[1])),
                                   value=0.))
        pad_heatmap_list.append(F.pad(heatmap,
                                      (0, int(max_w // 4 - heatmap.shape[2]), 0, int(max_h // 4 - heatmap.shape[1])),
                                      value=0.))
    max_num = 0
    for i in range(batch_size):
        n = boxes_list[i].shape[0]
        if n > max_num:
            max_num = n
    for i in range(batch_size):
        pad_boxes_list.append(F.pad(boxes_list[i], (0, 0, 0, max_num-boxes_list[i].shape[0]), value=-1))
        pad_class_list.append(F.pad(class_list[i], (0, max_num-class_list[i].shape[0]), value=-1))

    batch_boxes = torch.stack(pad_boxes_list)
    batch_classes = torch.stack(pad_class_list)
    batch_imgs = torch.stack(pad_imgs_list)
    batch_heatmaps = torch.stack(pad_heatmap_list)
    return batch_imgs, batch_boxes, batch_classes, batch_heatmaps, infos


class DataAug:
    def __init__(self, box_format='coco'):
        self.tsfm = A.Compose([
            A.HorizontalFlip(),
            # A.RandomResizedCrop(512, 512, scale=(0.75, 1)),
            A.RandomBrightnessContrast(0.4, 0.4),
            A.GaussNoise(),
            A.RGBShift(),
            A.CLAHE(),
            A.RandomGamma()
        ], bbox_params=A.BboxParams(format=box_format, min_visibility=0.75, label_fields=['labels']))

    def __call__(self, img, boxes, labels):
        augmented = self.tsfm(image=img, bboxes=boxes, labels=labels)
        img, boxes = augmented['image'], augmented['bboxes']
        return img, boxes


if __name__ == '__main__':
    r = gaussian_radius((20, 32))
    diameter = 2*r
    gauss = gaussian2D((diameter, diameter), sigma=diameter/6)
    print(gauss, gauss.shape)
