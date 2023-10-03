'''
Author: Chris Xiao yl.xiao@mail.utoronto.ca
Date: 2023-09-30 16:14:13
LastEditors: Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime: 2023-10-02 20:56:30
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
import json


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
    test_exp_mask_path = os.path.join(test_exp_path,'mask')
    test_exp_overlay_path = os.path.join(test_exp_path, 'overlay')
    
    make_if_dont_exist(test_exp_path)
    make_if_dont_exist(test_exp_mask_path)
    make_if_dont_exist(test_exp_overlay_path)
    
    test_dataset = EndoVisDataset(root_dir, ann_format=ann_format, img_format=img_format, mode='test', encoder_size=cfg.model.encoder_size)
    test_loader = DataLoader(test_dataset, batch_size=cfg.test_bs, shuffle=False, num_workers=cfg.num_workers)
    
    sam_mask_encoder, sam_prompt_encoder, sam_mask_decoder = sam_model_registry[cfg.model.sam_model_type](checkpoint=cfg.model.sam_model_dir,customized=cfg.model.sam_model_customized)
    model = EndoSAMAdapter(device, cfg.model.class_num, sam_mask_encoder, sam_prompt_encoder, sam_mask_decoder, num_token=cfg.num_token).to(device)
    weights = torch.load(os.path.join(model_exp_path,'model.pth'), map_location=device)['endosam_state_dict']
    model.load_state_dict(weights)
    
    model.eval()
    
    iou_dict = {}
    ious = []
    with torch.no_grad():
        for img, ann, name, img_bgr in test_loader:
            cv2.destroyAllWindows()
            img = img.to(device)
            ann = ann.to(device).unsqueeze(1).long()
            ann = one_hot_embedding_3d(ann, class_num=cfg.model.class_num)
            pred, pred_quality = model(img)
            mask_iou = np.nan
            if torch.unique(pred).size()[0] > 1:
                iou = jaccard(ann, pred)
                mask_iou = iou.item()
            iou_dict[name[0]] = mask_iou
            ious.append(mask_iou)
            pred = torch.argmax(pred, dim=1)
            numpy_pred = pred.squeeze(0).detach().cpu().numpy()
            numpy_pred[numpy_pred != 0] = 255
            img_bgr = img_bgr.squeeze(0).detach().cpu().numpy()
            # 将预测结果转换为三通道图像
            overlay = np.zeros_like(img_bgr)
            red_color = (0, 0, 255)  # 红色
            overlay[:,:,2][numpy_pred == 255] = 255
            # 将红色区域叠加在原图上
            alpha = 0.5  # 半透明度
            result = cv2.addWeighted(img_bgr, 1 - alpha, overlay, alpha, 0)
            cv2.imshow('Result', result)
            # 等待键盘输入（最多等待1秒）
            key = cv2.waitKey(1000)  # 超时时间为1000毫秒（1秒）
            # 判断是否有键盘输入
            if key == ord('q'):  # 如果用户按下 'q' 键
                cv2.destroyAllWindows()  # 关闭窗口
            else:
                # 继续执行其他操作
                pass
            cv2.imwrite(os.path.join(test_exp_mask_path, f'{name[0]}.png'), numpy_pred.astype(np.uint8))
            cv2.imwrite(os.path.join(test_exp_overlay_path, f'{name[0]}.png'), result)
    
    with open(os.path.join(test_exp_path, 'mask_ious.json'), 'w') as f:
        json.dump(iou_dict, f, indent=4, sort_keys=False)
    
    f.close()
    avg_iou = np.mean(ious, axis=0)
    print(f'average intersection over union of mask: {avg_iou}')