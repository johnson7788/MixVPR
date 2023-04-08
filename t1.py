#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2023/4/8 18:41
# @File  : t1.py
# @Author: 
# @Desc  :
import torch
from main import VPRModel

# Note that images must be resized to 320x320
model = VPRModel(backbone_arch='resnet50',
                 layers_to_crop=[4],
                 agg_arch='MixVPR',
                 agg_config={'in_channels' : 1024,
                             'in_h' : 20,
                             'in_w' : 20,
                             'out_channels' : 1024,
                             'mix_depth' : 4,
                             'mlp_ratio' : 1,
                             'out_rows' : 4},
                )

state_dict = torch.load('./checkpoint/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt')
model.load_state_dict(state_dict)
model.eval()