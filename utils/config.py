# -*- coding:utf-8 -*-
"""
Author:xufei
Date:2021/4/28
"""
import os
import yaml
from easydict import EasyDict as edict
base_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), '../'))
params_path = f'{base_dir}/centernet_resnet.yaml'


def config():
    with open(params_path, 'r', encoding='utf-8') as f:
        param = yaml.load(f, Loader=yaml.FullLoader)
        param = edict(param)
        # print(param)
    return param


if __name__ == '__main__':
    pass
