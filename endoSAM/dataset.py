'''
Author: Chris Xiao yl.xiao@mail.utoronto.ca
Date: 2023-09-16 17:41:29
LastEditors: Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime: 2023-10-02 20:52:54
FilePath: /EndoSAM/endoSAM/dataset.py
Description: EndoVisDataset class
I Love IU
Copyright (c) 2023 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved. 
'''
from torch.utils.data import Dataset
import os 
import glob
import re 
import numpy as np 
import cv2
from utils import ResizeLongestSide, preprocess
import torch

modes = ['train', 'val', 'test']

class EndoVisDataset(Dataset):
    def __init__(self, root, 
                 ann_format= 'png', 
                 img_format = 'jpg', 
                 mode='train',
                 encoder_size=1024):
        super(EndoVisDataset, self).__init__()
        """Define the customized EndoVis dataset

        Args:
            data_root_dir (str, optional): root dir containing all data. Defaults to "../data".
            mode (str, optional): either in "train", "val" or "test" mode. Defaults to "train".
            vit_mode (str, optional): "h", "l", "b" for huge, large, and base versions of SAM. Defaults to "h".
        """
        self.root = root
        self.mode = mode
        self.ann_format = ann_format
        self.img_format = img_format
        self.encoder_size = encoder_size
        self.ann_path = os.path.join(self.root, 'ann')
        self.img_path = os.path.join(self.root, 'img')
        
        if self.mode in modes:
            self.img_mode_path = os.path.join(self.img_path, self.mode)
            self.ann_mode_path = os.path.join(self.ann_path, self.mode)
        else:
            raise ValueError('Invalid mode: {}'.format(self.mode))
        
        self.imgs = glob.glob(os.path.join(self.img_mode_path, '*.{}'.format(self.img_format)))
        self.anns = glob.glob(os.path.join(self.ann_mode_path, '*.{}'.format(self.ann_format)))
        self.transform = ResizeLongestSide(self.encoder_size)
        
    def __len__(self):
        if self.mode in modes:
            assert len(self.imgs) == len(self.anns)
            return len(self.imgs)
        else:
            raise ValueError('Invalid mode: {}'.format(self.mode))
    
    def __getitem__(self, index) -> tuple:
        img_bgr = cv2.imread(self.imgs[index])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        name = os.path.basename(self.imgs[index]).split('.')[0]
        input_image = self.transform.apply_image(img_rgb)
        input_image_torch = torch.as_tensor(input_image).permute(2, 0, 1).contiguous()
        img = preprocess(input_image_torch, self.encoder_size)
        ann_path = os.path.join(self.ann_mode_path, f"{name}.{self.ann_format}")
        ann = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
        ann = np.array(ann)
        ann[ann != 0] = 1
        
        return img, ann, name, img_bgr