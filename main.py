# -*- coding:utf-8 -*-
"""
Author:xufei
Date:2021/4/27
"""
import random
import os
import torch
import numpy as np
from utils import Loss, Trainer
from libs.dataset import VOCDataset, collate_fn
from torch.utils.data import DataLoader
from libs import CenterNet
from utils.config import config
cfg = config()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(cfg.MODEL.seed)


def main(cfg):
    dataset = VOCDataset(cfg, resize=cfg.DATA.resize, mode=cfg.DATA.data_mode)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=cfg.MODEL.BATCH_SIZE,
                              shuffle=True,
                              num_workers=cfg.MODEL.NUM_WORKERS,
                              collate_fn=collate_fn,
                              pin_memory=True)
    model = CenterNet.CenterNet(cfg)
    if cfg.MODEL.device:
        model = model.cuda()
    loss_fn = Loss(cfg)
    cfg.MODEL.max_iter = len(train_loader)*cfg.MODEL.EPOCHS
    cfg.MODEL.steps = (int(cfg.MODEL.max_iter*0.6), int(cfg.MODEL.max_iter*0.8))
    train = Trainer(cfg, model, loss_fn, train_loader)
    train.forward()


if __name__ == '__main__':
    main(cfg)
