'''
Author: Chris Xiao yl.xiao@mail.utoronto.ca
Date: 2023-09-11 18:27:02
LastEditors: Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime: 2023-09-19 14:23:23
FilePath: /EndoSAM/endoSAM/train.py
Description: fine-tune training script
I Love IU
Copyright (c) 2023 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved. 
'''
'''
@copyright Chris Xiao yl.xiao@mail.utoronto.ca
'''
import argparse
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import os
from dataset import EndoVisDataset
from utils import make_if_dont_exist, setup_logger, one_hot_embedding_3d, save_checkpoint, plot_progress
import datetime
import torch
from model import EndoSAMAdapter
import numpy as np
from segment_anything.build_sam import sam_model_registry
from loss import ce_loss, mse_loss, jaccard


def parse_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None, type=str, help='path to config file')
    parser.add_argument('--resume', action='store_true', help='use this if you want to continue a training')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_command()
    cfg = args.cfg
    resume = args.resume
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    ckpt_path = os.path.join(cfg.ckpt_folder)
    plot_path = os.path.join(cfg.plot_folder)
    model_exp_path = os.path.join(model_path, exp)
    log_exp_path = os.path.join(log_path, exp)
    ckpt_exp_path = os.path.join(ckpt_path, exp)
    plot_exp_path = os.path.join(plot_path, exp)
    
    if not resume:
        make_if_dont_exist(model_path, overwrite=True)
        make_if_dont_exist(log_path, overwrite=True)
        make_if_dont_exist(ckpt_path, overwrite=True)
        make_if_dont_exist(plot_path, overwrite=True)
        make_if_dont_exist(model_exp_path, overwrite=True)
        make_if_dont_exist(log_exp_path, overwrite=True)
        make_if_dont_exist(ckpt_exp_path, overwrite=True)
        make_if_dont_exist(plot_exp_path, overwrite=True)
    
    datetime_object = 'training_log_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.log'
    logger = setup_logger(f'EndoSAM', os.path.join(log_exp_path, datetime_object))
    logger.info(f"Welcome To {exp} Fine-Tuning")
    
    logger.info("Load Dataset-Specific Parameters")
    train_dataset = EndoVisDataset(root_dir, ann_format=ann_format, img_format=img_format, mode='train', encoder_size=cfg.model.encoder_size)
    valid_dataset = EndoVisDataset(root_dir, ann_format=ann_format, img_format=img_format, mode='val', encoder_size=cfg.model.encoder_size)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train_bs, shuffle=True, num_workers=cfg.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.val_bs, shuffle=True, num_workers=cfg.num_workers)

    logger.info("Load Model-Specific Parameters")
    sam_mask_encoder, sam_prompt_encoder, sam_mask_decoder = sam_model_registry[cfg.model.model_type](checkpoint=f'../ckpt/sam/{cfg.model.model_name}',customized=cfg.model.model_customized)
    model = EndoSAMAdapter(device, cfg.model.class_num, sam_mask_encoder, sam_prompt_encoder, sam_mask_decoder, num_token=cfg.num_token).to(device)
    lr = cfg.opt_params.lr_default
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_values = []
    best_val_iou = -np.inf
    max_iter = cfg.max_iter
    val_iter = cfg.val_iter
    start_epoch = 0
    if resume:
        ckpt = torch.load(os.path.join(ckpt_exp_path, 'ckpt.pth'), map_location=device)
        optimizer.load_state_dict(ckpt['optimizer'])
        model.load_state_dict(ckpt['weights'])
        best_val_iou = ckpt['best_val_iou']
        train_losses = ckpt['train_losses']
        val_values = ckpt['val_values']
        lr = optimizer.param_groups[0]['lr']
        start_epoch = ckpt['epoch'] + 1
        logger.info("Resume Training")
    else:
        logger.info("Start Training")
    
    for epoch in range(start_epoch, cfg.max_iter):
        logger.info(f"Epoch {epoch+1}/{cfg.max_iter}:")
        losses = []
        model.train()
        for img, ann in train_loader:
            img = img.to(device)
            ann = ann.to(device).long()
            ann = one_hot_embedding_3d(ann, class_num=cfg.model.class_num+1)
            optimizer.zero_grad()
            pred, pred_quality = model(img)
            loss = cfg.losses.ce.weight * ce_loss(ann, pred) + cfg.losses.mse.weight * mse_loss(ann, pred)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = np.mean(losses, axis=0)
        logger.info(f"\ttraining loss: {avg_loss}")
        train_losses.append([epoch+1, avg_loss])
        
        if epoch % cfg.val_iter == 0:
            model.eval()
            ious = []
            with torch.no_grad():
                for img, ann in valid_loader:
                    img = img.to(device)
                    ann = ann.to(device).long()
                    ann = one_hot_embedding_3d(ann, class_num=cfg.model.class_num+1)
                    pred, pred_quality = model(img)
                    iou = jaccard(ann, pred)
                    ious.append(iou.item())
            
            avg_iou = np.mean(ious, axis=0)
            logger.info(f"\tvalidation iou: {avg_iou}")
            val_values.append([epoch+1, avg_iou])
            if avg_iou > best_val_iou:
                best_val_iou = avg_iou
                logger.info(f"\tsave best endosam model")
                torch.save({
                    'epoch': epoch,
                    'best_val_iou': best_val_iou,
                    'train_losses': train_losses,
                    'val_values': val_values,
                    'endosam_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(model_exp_path, 'model.pth'))
        save_dir = os.path.join(ckpt_exp_path, 'ckpt.pth')
        save_checkpoint(model, optimizer, epoch, best_val_iou, train_losses, val_values, save_dir)
        plot_progress(logger, plot_exp_path, train_losses, val_values, 'loss')
    
    
    
    
    

    
