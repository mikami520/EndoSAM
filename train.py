'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-09-11 18:27:02
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-09-12 21:57:50
FilePath: /finetune-anything/train.py
Description: 
I Love IU
Copyright (c) 2023 by error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git, All Rights Reserved. 
'''
'''
@copyright ziqi-jin
'''
import argparse
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from datasets import get_dataset
from losses import get_losses
from extend_sam import get_model, get_optimizer, get_scheduler, get_opt_pamams, get_runner

supported_tasks = ['detection', 'semantic_seg', 'instance_seg']
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', default='semantic_seg', type=str)
parser.add_argument('--cfg', default=None, type=str)
parser.add_argument('--low_rank', type=int)

if __name__ == '__main__':
    args = parser.parse_args()
    low_rank = args.low_rank
    task_name = args.task_name
    if args.cfg is not None:
        config = OmegaConf.load(args.cfg)
    else:
        assert task_name in supported_tasks, "Please input the supported task name."
        config = OmegaConf.load("./config/{task_name}.yaml".format(task_name=args.task_name))

    train_cfg = config.train
    val_cfg = config.val
    test_cfg = config.test

    train_dataset = get_dataset(train_cfg.dataset)
    train_loader = DataLoader(train_dataset, batch_size=train_cfg.bs, shuffle=True, num_workers=train_cfg.num_workers,
                              drop_last=train_cfg.drop_last)
    val_dataset = get_dataset(val_cfg.dataset)
    val_loader = DataLoader(val_dataset, batch_size=val_cfg.bs, shuffle=False, num_workers=val_cfg.num_workers,
                            drop_last=val_cfg.drop_last)
    losses = get_losses(losses=train_cfg.losses)
    # according the model name to get the adapted model
    model = get_model(model_name=train_cfg.model.sam_name, **train_cfg.model.params)
    opt_params = get_opt_pamams(model, lr_list=train_cfg.opt_params.lr_list, group_keys=train_cfg.opt_params.group_keys,
                                wd_list=train_cfg.opt_params.wd_list)
    optimizer = get_optimizer(opt_name=train_cfg.opt_name, params=opt_params, lr=train_cfg.opt_params.lr_default,
                              momentum=train_cfg.opt_params.momentum, weight_decay=train_cfg.opt_params.wd_default)
    scheduler = get_scheduler(optimizer=optimizer, lr_scheduler=train_cfg.scheduler_name)
    runner = get_runner(train_cfg.runner_name)(model, optimizer, losses, train_loader, val_loader, scheduler)
    # train_step
    runner.train(train_cfg)
    if test_cfg.need_test:
        runner.test(test_cfg)
