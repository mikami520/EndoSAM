'''
Author: Chris Xiao yl.xiao@mail.utoronto.ca
Date: 2023-09-11 18:27:02
LastEditors: Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime: 2023-12-12 16:18:58
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
from loss import ce_loss, mse_loss
from tqdm import tqdm


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
    model_path = cfg.model_folder
    log_path = cfg.log_folder
    ckpt_path = cfg.ckpt_folder
    plot_path = cfg.plot_folder
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
    sam_mask_encoder, sam_prompt_encoder, sam_mask_decoder = sam_model_registry[cfg.model.sam_model_type](checkpoint=cfg.model.sam_model_dir,customized=cfg.model.sam_model_customized)
    model = EndoSAMAdapter(device, cfg.model.class_num, sam_mask_encoder, sam_prompt_encoder, sam_mask_decoder, num_token=cfg.num_token).to(device)
    lr = cfg.opt_params.lr_default
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    best_val_loss = np.inf
    max_iter = cfg.max_iter
    val_iter = cfg.val_iter
    start_epoch = 0
    if resume:
        ckpt = torch.load(os.path.join(ckpt_exp_path, 'ckpt.pth'), map_location=device)
        optimizer.load_state_dict(ckpt['optimizer'])
        model.load_state_dict(ckpt['weights'])
        best_val_loss = ckpt['best_val_loss']
        train_losses = ckpt['train_losses']
        val_losses = ckpt['val_losses']
        lr = optimizer.param_groups[0]['lr']
        start_epoch = ckpt['epoch'] + 1
        logger.info("Resume Training")
    else:
        logger.info("Start Training")
    
    for epoch in range(start_epoch, cfg.max_iter):
        logger.info(f"Epoch {epoch+1}/{cfg.max_iter}:")
        losses = []
        model.train()
        with tqdm(train_loader, unit='batch', desc='Training') as tdata:
            for img, ann, _, _ in tdata:
                img = img.to(device)
                ann = ann.to(device).unsqueeze(1).long()
                ann = one_hot_embedding_3d(ann, class_num=cfg.model.class_num)
                optimizer.zero_grad()
                pred, pred_quality = model(img)
                loss = cfg.losses.ce.weight * ce_loss(ann, pred) + cfg.losses.mse.weight * mse_loss(ann, pred)
                tdata.set_postfix(loss=loss.item())
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        avg_loss = np.mean(losses, axis=0)
        logger.info(f"\ttraining loss: {avg_loss}")
        train_losses.append([epoch+1, avg_loss])
        
        if epoch % cfg.val_iter == 0:
            model.eval()
            losses = []
            with torch.no_grad():
                with tqdm(valid_loader, unit='batch', desc='Validation') as tdata:
                    for img, ann, _, _ in tdata:
                        img = img.to(device)
                        ann = ann.to(device).unsqueeze(1).long()
                        ann = one_hot_embedding_3d(ann, class_num=cfg.model.class_num)
                        pred, pred_quality = model(img)
                        loss = cfg.losses.ce.weight * ce_loss(ann, pred) + cfg.losses.mse.weight * mse_loss(ann, pred)
                        tdata.set_postfix(loss=loss.item())
                        losses.append(loss.item())
            
            avg_loss = np.mean(losses, axis=0)
            logger.info(f"\tvalidation loss: {avg_loss}")
            val_losses.append([epoch+1, avg_loss])
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                logger.info(f"\tsave best endosam model")
                torch.save({
                    'epoch': epoch,
                    'best_val_loss': best_val_loss,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'endosam_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(model_exp_path, 'model.pth'))
        save_dir = os.path.join(ckpt_exp_path, 'ckpt.pth')
        save_checkpoint(model, optimizer, epoch, best_val_loss, train_losses, val_losses, save_dir)
        plot_progress(logger, plot_exp_path, train_losses, val_losses, 'loss')
    
    
    
    
    

    
