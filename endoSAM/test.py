'''
Author: Chris Xiao yl.xiao@mail.utoronto.ca
Date: 2023-09-30 16:14:13
LastEditors: mikami520 yl.xiao@mail.utoronto.ca
LastEditTime: 2023-09-30 23:44:34
FilePath: /EndoSAM/endoSAM/test.py
Description: fine-tune inference script
I Love IU
Copyright (c) 2023 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved. 
'''
import argparse
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import os
from dataset import EndoVisDataset
from utils import make_if_dont_exist, one_hot_embedding_3d
import torch
from model import EndoSAMAdapter
import numpy as np
from segment_anything.build_sam import sam_model_registry
from loss import jaccard
import cv2


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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    exp = cfg.experiment_name
    root_dir = cfg.dataset.dataset_dir
    img_format = cfg.dataset.img_format
    ann_format = cfg.dataset.ann_format
    model_path = cfg.model_folder
    model_exp_path = os.path.join(model_path, exp)
    test_path = cfg.test_folder
    test_exp_path = os.path.join(test_path, exp)
    
    make_if_dont_exist(test_exp_path)
    
    test_dataset = EndoVisDataset(root_dir, ann_format=ann_format, img_format=img_format, mode='test', encoder_size=cfg.model.encoder_size)
    test_loader = DataLoader(test_dataset, batch_size=cfg.test_bs, shuffle=False, num_workers=cfg.num_workers)
    
    sam_mask_encoder, sam_prompt_encoder, sam_mask_decoder = sam_model_registry[cfg.model.sam_model_type](checkpoint=cfg.model.sam_model_dir,customized=cfg.model.sam_model_customized)
    model = EndoSAMAdapter(device, cfg.model.class_num, sam_mask_encoder, sam_prompt_encoder, sam_mask_decoder, num_token=cfg.num_token).to(device)
    weights = torch.load(os.path.join(model_exp_path,'model.pth'), map_location=device)['endosam_state_dict']
    model.load_state_dict(weights)
    
    model.eval()
    
    ious = []
    with torch.no_grad():
        for img, ann, name, img_rgb in test_loader:
            cv2.destroyAllWindows()
            img = img.to(device)
            ann = ann.to(device).unsqueeze(1).long()
            ann = one_hot_embedding_3d(ann, class_num=cfg.model.class_num)
            pred, pred_quality = model(img)
            iou = jaccard(ann, pred)
            ious.append(iou.item())
            pred = torch.argmax(pred, dim=1)
            numpy_pred = pred.cpu().detach().numpy().squeeze()
            numpy_pred[numpy_pred != 0] = 255
            img_rgb = img_rgb.detach().cpu().numpy().squeeze()
            # 将预测结果转换为三通道图像
            overlay = np.zeros_like(img_rgb)
            red_color = (0, 0, 255)  # 红色
            overlay[:,:,2][numpy_pred == 255] = 255
            # 将红色区域叠加在原图上
            alpha = 0.5  # 半透明度
            result = cv2.addWeighted(img_rgb, 1 - alpha, overlay, alpha, 0)
            cv2.imshow('Result', result)
            # 等待键盘输入（最多等待1秒）
            key = cv2.waitKey(1000)  # 超时时间为1000毫秒（1秒）
            # 判断是否有键盘输入
            if key == ord('q'):  # 如果用户按下 'q' 键
                cv2.destroyAllWindows()  # 关闭窗口
            else:
                # 继续执行其他操作
                pass
            # cv2.imwrite(os.path.join(test_exp_path, f'{name[0]}.png'), numpy_pred.astype(np.uint8))
            cv2.imwrite(os.path.join(test_exp_path, f'{name[0]}.png'), result)
    
    avg_iou = np.mean(ious, axis=0)
    print(f'average intersection over union of mask: {avg_iou}')