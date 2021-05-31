# -*- coding:utf-8 -*-
"""
Author:xufei
Date:2021/5/7
"""
import os
import torch
import numpy as np
import torchvision.transforms as T
from libs.network import CenterNet
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from utils.config import config
cfg = config()
test_path = './data/000009.jpg'


def preprocess_img(img, input_ksize):
    min_side, max_side = input_ksize
    h, w = img.height, img.width
    _pad = 32
    smalle_side = min(w, h)
    larget_side = max(w, h)
    scale = min_side / smalle_side
    if larget_side * scale > max_side:
        scale = max_side / larget_side
    new_w, new_h = int(scale*w), int(scale*h)
    img_resize = np.array(img.resize((new_w, new_h)))
    pad_w, pad_h = _pad-new_w%_pad, _pad-new_h%_pad
    img_pad = np.zeros(shape=[new_h+pad_h, new_w+pad_w, 3], dtype=np.uint8)
    img_pad[:new_h, :new_w, :] = img_resize
    return img_pad, {'raw_h': h, 'raw_w': w}


def show_img(img, boxes, classes, scores):
    boxes, scores = [i.cpu() for i in [boxes, scores]]
    boxes = boxes.long()
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle(xy=box.tolist(), outline='red')
        draw.rectangle(xy=(box+1).tolist(), outline='blue')
        draw.rectangle(xy=(box+2).tolist(), outline='green')
    img.show()
    # boxes = boxes.tolist()
    # scores = scores.tolist()
    # plt.figure(figsize=(10, 10))
    # for i in range(len(boxes)):
    #     plt.text(x=boxes[i][0], y=boxes[i][1],
    #              s='{}:{:.4f}'.format(classes[i], scores[i]),
    #              wrap=True, size=15, bbox=dict(facecolor='r', alpha=0.7))
    #     plt.imshow(img)
    #     plt.show()


def main():
    # 模型加载
    checkpoint_path = os.path.join(cfg.OUTPUT.checkpoint_dir, '2021-05-07', 'best_model_checkpoint.pth')
    checkpoint = torch.load(checkpoint_path)

    # 图片前处理
    img = Image.open(test_path).convert('RGB')
    img_pad, info = preprocess_img(img, cfg.DATA.resize)
    imgs, infos = [img], [info]
    transform = T.Compose([T.ToTensor(), T.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)])
    inputs = transform(img_pad).unsqueeze(0).cuda()

    # 图像前向处理
    model = CenterNet.CenterNet(cfg).cuda()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    detects = model.inference(inputs, topk=40, return_hm=False, theta=0.1)

    for img_idx in range(len(detects)):
        boxes = detects[img_idx][0]
        scores = detects[img_idx][1]
        classes = detects[img_idx][2]
        img = imgs[img_idx]
        show_img(img, boxes, classes, scores)


if __name__ == '__main__':
    main()
