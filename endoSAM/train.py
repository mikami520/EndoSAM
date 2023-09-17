'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-09-11 18:27:02
LastEditors: Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime: 2023-09-16 22:17:57
FilePath: /EndoSAM/endoSAM/train.py
Description: fine-tune training script
I Love IU
Copyright (c) 2023 by error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git, All Rights Reserved. 
'''
'''
@copyright Chris Xiao yl.xiao@mail.utoronto.ca
'''
import argparse
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import os
from dataset import EndoVisDataset
from utils import make_if_dont_exist, setup_logger
import datetime
import torch
from model import EndoSAMAdapter, Learnable_Prototypes
import numpy as np
from segment_anything.build_sam import sam_model_registry


def parse_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None, type=str, help='path to config file')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_command()
    cfg = args.cfg
    if cfg is not None:
        if os.path.exists(cfg):
            cfg = OmegaConf.load(cfg)
        else:
            raise FileNotFoundError(f'config file {cfg} not found')
    else:
        raise ValueError('config file not specified')

    exp = cfg.experiment_name
    root_dir = cfg.dataset.dataset_dir
    img_format = cfg.dataset.img_format
    ann_format = cfg.dataset.ann_format
    model_path = os.path.join(cfg.model_folder)
    log_path = os.path.join(cfg.log_folder)
    model_exp_path = os.path.join(model_path, exp)
    log_exp_path = os.path.join(log_path, exp)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    make_if_dont_exist(model_path)
    make_if_dont_exist(log_path)
    make_if_dont_exist(model_exp_path)
    make_if_dont_exist(log_exp_path)
    
    datetime_object = 'training_log_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.log'
    logger = setup_logger(f'EndoSAM', os.path.join(log_exp_path, datetime_object))
    logger.info(f"======> Welcome To {exp} Fine-Tuning")
    
    logger.info("======> Load Dataset-Specific Parameters")
    train_dataset = EndoVisDataset(root_dir, ann_format=ann_format, img_format=img_format, mode='train', encoder_size=cfg.model.encoder_size)
    valid_dataset = EndoVisDataset(root_dir, ann_format=ann_format, img_format=img_format, mode='val', encoder_size=cfg.model.encoder_size)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train_bs, shuffle=True, num_workers=cfg.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.val_bs, shuffle=True, num_workers=cfg.num_workers)

    logger.info("======> Load Model-Specific Parameters")
    sam_mask_encoder, sam_prompt_encoder, sam_mask_decoder = sam_model_registry[cfg.model.model_type](checkpoint=f'../ckpt/sam/{cfg.model.model_name}',customized=cfg.model.model_customized)
    model = EndoSAMAdapter(device, cfg.model.class_num, sam_mask_encoder, sam_prompt_encoder, sam_mask_decoder, num_token=cfg.num_token)
    learnable_prototypes_model = Learnable_Prototypes(num_classes=cfg.model.class_num-1, feat_dim = 256).to(device)
    prototypes = learnable_prototypes_model()
    lr = cfg.opt_params.lr_default
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loss = []
    val_loss = []
    best_val_loss = np.inf
    max_iter = cfg.max_iter
    val_iter = cfg.val_iter
    
    logger.info("======> Start Training")
    for epoch in range(cfg.max_iter):
        for img, ann in train_loader:
            img = img.to(device)
            ann = ann.to(device)
            optimizer.zero_grad()
            pred, pred_quality = model(prototypes, img)
            print(pred.shape, pred_quality.shape)
    
    
    
    
    
    

    
