# -*- coding:utf-8 -*-
"""
Author:xufei
Date:2021/4/29
"""
import torch
import os, cv2
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
import torchvision.transforms as T
from libs.dataset.tools import *
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoDetection


class VOCDataset(Dataset):
    CLASSES_NAME = ('__back_ground__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self, cfg, resize=(512, 512), mode='train', img_transform=None):
        self.root = cfg.DATA.ROOT
        self.img_path = os.path.join(self.root, 'JPEGImages')
        self.ann = os.path.join(self.root, 'Annotations')
        self.mode = mode
        self.img_transform = img_transform
        self.transform = T.Compose([T.ToTensor(), T.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)])
        self.resize = resize
        self.down_stride = cfg.DATA.down_stride
        self.category2id = {k: v for v, k in enumerate(self.CLASSES_NAME)}
        self.id2category = {v: k for k, v in self.category2id.items()}
        with open(os.path.join(self.root, 'ImageSets', 'Main', f'{self.mode}.txt'), 'r', encoding='utf-8') as f:
            self.samples = f.read().splitlines()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name = self.samples[idx] + '.jpg'
        img = Image.open(os.path.join(self.img_path, img_name)).convert('RGB')
        img = np.array(img)
        raw_h, raw_w, _ = img.shape
        info = {'raw_h': raw_h, 'raw_w': raw_w}
        target_name = self.samples[idx] + '.xml'
        label = self._xml_parse(os.path.join(self.ann, target_name))
        if self.mode == 'train' and self.img_transform:
            img, boxes = self.img_transform(img, label['boxes'], label['labels'])
        else:
            boxes = np.array(label['boxes']) if label['boxes'] else np.array([[0, 0, 0, 0]])

        boxes_w, boxes_h = boxes[..., 2] - boxes[..., 0], boxes[..., 3] - boxes[..., 1]
        center = np.array([(boxes[..., 0] + boxes[..., 2]) / 2, (boxes[..., 1] + boxes[..., 3]) / 2],
                          dtype=np.float32).T
        img, boxes = self._precess_img_boxes(img, self.resize, boxes, img_name)
        info['resize_h'], info['resize_w'] = img.shape[:2]
        classes = label['labels']
        img = self.transform(img)
        boxes = torch.from_numpy(boxes).float()
        classes = torch.LongTensor(classes)

        out_h, out_w = info['resize_h'] // self.down_stride, info['resize_w'] // self.down_stride
        boxes_h, boxes_w, center = boxes_h / self.down_stride, boxes_w / self.down_stride, center / self.down_stride
        heatmap = np.zeros((20, out_h, out_w), dtype=np.float32)
        center[:, 0] = np.clip(center[:, 0], 0, out_w - 1)
        center[:, 1] = np.clip(center[:, 1], 0, out_h - 1)
        info['gt_heatmap_h'], info['gt_heatmap_w'] = out_h, out_w
        obj_mask = torch.ones(len(classes))
        for i, cls_id in enumerate(classes):
            radius = gaussian_radius((np.ceil(boxes_h[i]), np.ceil(boxes_w[i])))
            radius = max(0, int(radius))
            center_int = center[i].astype(np.int32)
            if (heatmap[:, center_int[1], center_int[0]] == 1).sum() >= 1.:
                obj_mask[i] = 0
                continue
            draw_umich_gaussian(heatmap[cls_id-1], center_int, radius)
            if heatmap[cls_id-1, center_int[1], center_int[0]] != 1:
                obj_mask[i] = 0

        heatmap = torch.from_numpy(heatmap)
        obj_mask = obj_mask.eq(1)
        boxes = boxes[obj_mask]
        classes = classes[obj_mask]
        info['center'] = torch.tensor(center)[obj_mask]

        assert heatmap.eq(1).sum().item() == len(classes) == len(info['center']), \
            f'index:{img_name}, heatmap peer:{heatmap.eq(1).sum().item()}, obj num:{len(classes)}'
        # print(img.size(), boxes.size(), classes.size(), heatmap.size())
        return img, boxes, classes, heatmap, info

    @staticmethod
    def _precess_img_boxes(img, resize, boxes, img_name):
        min_side, max_side = resize
        h, w, _ = img.shape
        pad = 32
        small_side, large_side = min(h, w), max(h, w)
        scale = min_side / small_side
        if large_side * scale > max_side:
            scale = max_side / large_side
        new_w, new_h = int(scale * w), int(scale * h)
        img_resize = cv2.resize(img, (new_w, new_h))
        pad_w, pad_h = pad - new_w % pad, pad - new_h % pad
        image_pad = np.zeros((new_h + pad_h, new_w + pad_w, 3), dtype=np.uint8)
        image_pad[:new_h, :new_w, :] = img_resize
        if boxes.any():
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
            return image_pad, boxes
        else:
            print('error img', img_name)
            return image_pad, boxes

    def _xml_parse(self, ann_path):
        tree = ET.parse(ann_path)
        root = tree.getroot()
        objs = root.findall('object')

        boxes = list()
        labels = list()
        difficulties = list()
        for obj in objs:
            difficult = int(obj.find('difficult').text == '1')
            label = obj.find('name').text.lower().strip()
            if label not in self.CLASSES_NAME:
                continue
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text) - 1
            ymin = int(bbox.find('ymin').text) - 1
            xmax = int(bbox.find('xmax').text) - 1
            ymax = int(bbox.find('ymax').text) - 1
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.category2id[label])
            difficulties.append(difficult)
        return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}


class COCODataset(CocoDetection):
    CLASSES_NAME = (
        '__back_ground__', 'person', 'bicycle', 'car', 'motorcycle',
        'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
        'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle',
        'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli',
        'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet',
        'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush')

    def __init__(self, imgs_path, anno_path, resize_size=(800, 1024), mode='TRAIN', img_transform=None):
        super().__init__(imgs_path, anno_path)

        print("INFO====>check annos, filtering invalid data......")
        ids = []
        for id in self.ids:
            ann_id = self.coco.getAnnIds(imgIds=id, iscrowd=None)
            ann = self.coco.loadAnns(ann_id)
            if self._has_valid_annotation(ann):
                ids.append(id)
        self.ids = ids
        self.category2id = {v: i + 1 for i, v in enumerate(self.coco.getCatIds())}
        self.id2category = {v: k for k, v in self.category2id.items()}

        self.img_transform = img_transform(mode)
        self.transform = T.Compose([T.ToTensor(), T.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)])
        self.resize_size = resize_size
        self.down_stride = 4
        self.mode = mode.lower()

    def __getitem__(self, index):
        img, ann = super().__getitem__(index)
        # print(ann)
        info = {'index': index}

        ann = [o for o in ann if o['iscrowd'] == 0]
        boxes = [o['bbox'] for o in ann]
        boxes = np.array(boxes, dtype=np.float32)
        boxes_w, boxes_h = boxes[..., 2], boxes[..., 3]

        # xywh-->xyxy
        boxes[..., 2:] = boxes[..., 2:] + boxes[..., :2]
        # x, y
        ct = np.array([(boxes[..., 0] + boxes[..., 2]) / 2,
                       (boxes[..., 1] + boxes[..., 3]) / 2], dtype=np.float32).T
        img = np.array(img)
        h, w, _ = img.shape
        info['raw_height'], info['raw_width'] = h, w
        if self.mode == 'TRAIN':
            img, boxes = self.img_transform(img, boxes)

        img, boxes = self.preprocess_img_boxes(img, self.resize_size, boxes)
        info['resize_height'], info['resize_width'] = img.shape[:2]

        classes = [o['category_id'] for o in ann]
        classes = [self.category2id[c] for c in classes]

        img = self.transform(img)
        boxes = torch.from_numpy(boxes)
        classes = torch.LongTensor(classes)

        output_h, output_w = info['resize_height'] // self.down_stride, info['resize_width'] // self.down_stride
        boxes_h, boxes_w, ct = boxes_h / self.down_stride, boxes_w / self.down_stride, ct / self.down_stride
        hm = np.zeros((80, output_h, output_w), dtype=np.float32)
        ct[:, 0] = np.clip(ct[:, 0], 0, output_w - 1)
        ct[:, 1] = np.clip(ct[:, 1], 0, output_h - 1)
        info['gt_hm_height'], info['gt_hm_witdh'] = output_h, output_w
        obj_mask = torch.ones(len(classes))
        for i, cls_id in enumerate(classes):
            radius = gaussian_radius((np.ceil(boxes_h[i]), np.ceil(boxes_w[i])))
            radius = max(0, int(radius))
            ct_int = ct[i].astype(np.int32)
            if (hm[:, ct_int[1], ct_int[0]] == 1).sum() >= 1.:
                obj_mask[i] = 0
                continue

            draw_umich_gaussian(hm[cls_id - 1], ct_int, radius)
            if hm[cls_id-1, ct_int[1], ct_int[0]] != 1:
                obj_mask[i] = 0

        hm = torch.from_numpy(hm)
        obj_mask = obj_mask.eq(1)
        boxes = boxes[obj_mask]
        classes = classes[obj_mask]
        info['ct'] = torch.tensor(ct)[obj_mask]

        assert hm.eq(1).sum().item() == len(classes) == len(info['ct']), \
            f"index: {index}, hm peer: {hm.eq(1).sum().item()}, object num: {len(classes)}"
        return img, boxes, classes, hm, info

    @staticmethod
    def preprocess_img_boxes(image, input_ksize, boxes=None):
        '''
        resize image and bboxes
        Returns
        image_paded: input_ksize
        bboxes: [None,4]
        '''
        min_side, max_side = input_ksize
        h, w, _ = image.shape
        _pad = 32  # 32

        smallest_side = min(w, h)
        largest_side = max(w, h)
        scale = min_side / smallest_side
        if largest_side * scale > max_side:
            scale = max_side / largest_side
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        pad_w = _pad - nw % _pad
        pad_h = _pad - nh % _pad

        image_paded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
        image_paded[:nh, :nw, :] = image_resized

        if boxes is None:
            return image_paded
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
            return image_paded, boxes

    @staticmethod
    def _has_only_empty_bbox(annot):
        return all(any(o <= 1 for o in obj['bbox'][2:]) for obj in annot)

    def _has_valid_annotation(self, annot):
        if len(annot) == 0:
            return False
        if self._has_only_empty_bbox(annot):
            return False
        return True


class OwnDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return

    def __getitem__(self, idx):
        return


if __name__ == '__main__':
    from utils.config import config
    cfg = config()
    data = VOCDataset(cfg, resize=(512, 512))
    loader = DataLoader(data, batch_size=8, collate_fn=collate_fn)
    batch = next(iter(loader))
    # print(len(loader))
    import tqdm
    for i, data in enumerate(tqdm.tqdm(loader)):
        imgs, boxes, classes, hms, infos = data
    #     print(imgs.shape, boxes.shape, classes.shape, hms.shape)
    #     print(infos)
