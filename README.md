<!--
 * @Author: Chris Xiao yl.xiao@mail.utoronto.ca
 * @Date: 2023-09-12 22:10:18
 * @LastEditors: Chris Xiao yl.xiao@mail.utoronto.ca
 * @LastEditTime: 2023-09-30 17:26:27
 * @FilePath: /EndoSAM/README.md
 * @Description: 
 * I Love IU
 * Copyright (c) 2023 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved. 
-->
# EndoSAM
Fine-tune for endoscope clapster segmentation

## Installation (tested on Ubuntu 20.04.6 LTS x86_64)
```
git clone https://github.com/mikami520/EndoSAM.git
cd EndoSAM
conda env create -f environment.yaml
conda activate sam
```
If conda cannot install successfully, try
 ```
conda create -y -n sam python=3.10.11
conda activate sam
pip install -r requirements.txt
```

## Usage
- Download (using ```wget``` or manual way) the SAM model checkpoint and place it into ```ckpt/sam```
Click the links below to download the checkpoint for the corresponding model type.

    - **`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
    - `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
    - `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
- Run the script (change the config file for play)
```
cd endoSAM
python train.py --cfg ../config/finetune.yaml
```
- GPU RAM Requirement\
Even though this is the fine-tune work, it requires a large GPU RAM. **Use suitable batch size based on the VRAM you have**
    - Batch Size 1 -> 6 GB RAM
    - Batch Size 2 -> 12 GB RAM
    - Batch Size 4 -> 21 GB RAM
    - Batch Size 8 -> 33 GB RAM

The ```training checkpoints```,```best model```,```loss plots```and```log files``` will be saved in the```log_folder```,```model_folder```,```ckpt_folder``` and ```plot_folder```you provide in the config file.
## Inference
```
python test.py --cfg ../config/finetune.yaml
```
The prediction results will be saved into the ```test_folder``` you provide in the config file.