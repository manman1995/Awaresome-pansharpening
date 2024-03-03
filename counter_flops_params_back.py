#!/usr/bin/env python
# coding=utf-8
'''
Author: wjm
Date: 2021-03-20 14:44:14
LastEditTime: 2021-03-22 15:25:02
Description: file content
'''
from thop import profile
import importlib, torch
from utils.config import get_config
import math

if __name__ == "__main__":
    model_name = 'DBPN'
    net_name = model_name.lower()
    lib = lib = importlib.import_module('model.' + net_name)
    net = lib.Net
    cfg = get_config('option.yml')
    model = net(
            num_channels=3, 
            base_filter=64,  
            scale_factor=2,
            args= cfg
    )
    input = torch.randn(1, 3, 48, 48)
    flops, params = profile(model, inputs=(input,))
    print('flops:{:.2f}, params:{:.2f}'.format(flops/(1e9), params/(1e6)))