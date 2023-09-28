'''
Author: Chris Xiao yl.xiao@mail.utoronto.ca
Date: 2023-09-16 18:21:41
LastEditors: Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime: 2023-09-27 21:06:26
FilePath: /EndoSAM/endoSAM/loss.py
Description: loss functions
I Love IU
Copyright (c) 2023 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved. 
'''
import torch
import torch.nn as nn
from torchmetrics.classification import JaccardIndex

def mse_loss(gt, pred):
    mse = nn.MSELoss().to(pred.device)
    return mse(pred, gt)

def ce_loss(gt, pred):
    ce = nn.CrossEntropyLoss().to(pred.device)
    return ce(pred, gt)

def jaccard(gt, pred):
    jaccard = JaccardIndex(task='multiclass', num_classes=2, average='micro').to(pred.device)
    return jaccard(pred, gt)