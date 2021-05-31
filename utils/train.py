# -*- coding:utf-8 -*-
"""
Author:xufei
Date:2021/5/7
"""
import torch
import os, logging
from datetime import datetime
from utils.lr_scheduler import *
from torch.cuda.amp import GradScaler, autocast


class Trainer:
    def __init__(self, cfg, model, loss_fn, train_loader, val_loader=None):
        self.Scaler = GradScaler()      # 混合精度（amp）
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.loss = loss_fn
        self.device = cfg.MODEL.device if torch.cuda.is_available() else 'cpu'

        if cfg.MODEL.resume:
            self.resume_model()
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.MODEL.lr,
                                               weight_decay=cfg.MODEL.weight_decay,
                                               amsgrad=cfg.MODEL.amsgrad)
            self.lr_schedule = WarmupMultiStepLR(self.optimizer,
                                                 cfg.MODEL.steps,
                                                 gamma=cfg.MODEL.gamma,
                                                 warmup_iters=cfg.MODEL.warmup_iters)
            self.start_step = 1
            self.best_loss = 1e6

        if self.device == cfg.MODEL.device:
            self.model = self.model.cuda()
            self.loss = self.loss.cuda()

        # 模型保存路径
        if not os.path.exists(cfg.OUTPUT.log_dir):
            os.makedirs(cfg.OUTPUT.log_dir)
        if not os.path.exists(cfg.OUTPUT.checkpoint_dir):
            os.makedirs(cfg.OUTPUT.checkpoint_dir)

        self.cfg = cfg
        self.logger = self.init_logger(cfg)
        self.logger.info('Start training !')

    def forward(self):
        self.logger.info('Start training ... \n')
        while self.start_step < self.cfg.MODEL.max_iter:
            loss = self.train_one_epoch()
            if self.cfg.MODEL.EVAL:
                loss = self.val_one_epoch()
                self.write_log(loss, mode='EVAL')
            self.save_model(loss < self.best_loss)
            self.best_loss = min(self.best_loss, loss)

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        for step, dataset in enumerate(self.train_loader):
            # if step > 10:
            #     break
            if self.device:
                dataset = [data.cuda() if isinstance(data, torch.Tensor) else data for data in dataset]
            self.optimizer.zero_grad()

            # 半精度训练
            if self.cfg.MODEL.amp:
                with autocast():
                    pre = self.model(dataset[0])
                    losses = self.loss(pre, dataset)
                    loss = sum(losses)
                self.Scaler.scale(loss).backward()
                self.Scaler.step(self.optimizer)
                self.Scaler.update()
            else:
                pre = self.model(dataset[0])
                losses = self.loss(pre, dataset)
                loss = sum(losses)
                loss.backward()
                self.optimizer.step()
                self.lr_schedule.step()

            total_loss += loss.item()
            self.start_step += 1
            if step % self.cfg.OUTPUT.log_interval == 0:
                self.write_log(total_loss/(step+1), losses[0].item(), losses[1].item())
        return total_loss / (step + 1)

    @torch.no_grad()
    def val_one_epoch(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for step, dataset in enumerate(self.val_loader):
                if self.device:
                    dataset = [data.cuda() if isinstance(data, torch.Tensor) else data for data in dataset]
                pre = self.model(dataset[0])
                losses = self.loss(pre, dataset)
                total_loss += sum(losses).item()
        return total_loss / (step + 1)

    @staticmethod
    def init_logger(cfg):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        format = logging.Formatter('%(asctime)s - %(message)s')
        handler = logging.FileHandler(os.path.join(cfg.OUTPUT.log_dir, 'log.txt'))
        handler.setLevel(logging.INFO)
        handler.setFormatter(format)
        logger.addHandler(handler)

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(format)
        logger.addHandler(console)
        return logger

    @staticmethod
    def get_lr(optim):
        for param_group in optim.param_groups:
            return param_group['lr']

    def write_log(self, avg_loss, cls_loss=None, iou_loss=None, mode='TRAIN'):
        log = f'[{mode}] TOTAL STEP: %6d/{self.cfg.MODEL.max_iter}' % self.start_step
        if cls_loss is not None:
            log += f'\t cls loss: %.3f' % cls_loss
        if iou_loss is not None:
            log += f'\t iou loss: %.3f' % iou_loss
        log += f'\t avg loss: %.6f' % avg_loss
        log += f'\t lr: %.6f' % self.get_lr(self.optimizer)
        self.logger.info(log)

    def save_model(self, is_best=False):
        state_dict = {'model': self.model.state_dict(),
                      'step': self.start_step,
                      'optimizer': self.optimizer,
                      'lr_schedule': self.lr_schedule,
                      'loss': self.best_loss,
                      'config': self.cfg}
        checkpoints_dir = os.path.join(self.cfg.OUTPUT.checkpoint_dir, datetime.now().strftime('%Y-%m-%d'))
        if not os.path.exists(checkpoints_dir):
            os.mkdir(checkpoints_dir)
        if is_best:
            torch.save(state_dict, os.path.join(checkpoints_dir, 'best_model_checkpoint.pth'))
        torch.save(state_dict, os.path.join(checkpoints_dir, 'model_checkpoint.pth'))

    def resume_model(self):
        if self.cfg.OUTPUT.resume_from_best:
            path = os.path.join(self.cfg.OUTPUT.checkpoint_dir, 'best_model_checkpoint.pth')
        else:
            path = os.path.join(self.cfg.OUTPUT.checkpoint_dir, 'model_checkpoint.pth')
        checkpoint = torch.load(path)
        model_state_dict = checkpoint['model']
        self.optimizer = checkpoint['optimizer']
        self.lr_schedule = checkpoint['lr_sch']
        self.start_step = checkpoint['step']
        self.best_loss = checkpoint['loss']
        self.model.load_state_dict(model_state_dict)


if __name__ == '__main__':
    pass
