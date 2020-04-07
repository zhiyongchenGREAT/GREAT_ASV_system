#!/usr/bin/env python

import os
import numpy as np
import time
import logging
import sys
import shutil
from datetime import datetime

import torch
from torch.utils.data import DataLoader, RandomSampler
from my_dataloader import My_DataLoader
from read_data import CSVDataSet, WithReplacementRandomSampler, PickleDataSet, PickleDataSet_single
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from config.config import Config
from train_model_new import get_model
from tensorboardX import SummaryWriter
import utils

torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == '__main__':
    opt = Config()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    torch.backends.cudnn.benchmark = opt.cudnn_benchmark

    print('GPU ID:', opt.gpu_id)
    [print('CUDA_device:', torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = get_model(opt.model, opt.metric, 0, opt.model_settings, opt)

    if torch.cuda.is_available():
        print("Data Parallel on ", torch.cuda.device_count(), "GPUs!")
    else:
        print("No GPU available!")
        sys.exit(1)

    
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2, eta_min=1e-4, last_epoch=-1)

    train_data = PickleDataSet_single(opt.train_list)
    train_dataloader = My_DataLoader(train_data, batch_size=opt.train_batch_size, shuffle=False, sampler=RandomSampler(train_data, replacement=True),\
    batch_sampler=None, num_workers=opt.num_workers, collate_fn=None,\
    pin_memory=False, drop_last=False, timeout=0,\
    worker_init_fn=None, multiprocessing_context=None)

    model, optimizer, scheduler, total_step = resume_training(opt, model, optimizer, scheduler)

    opt = dir_init(opt)

    tbx_writer = tensorboard_init(opt)

    model = model.to(device)       

    timing_point = time.time()
    expected_total_step_epoch = get_epoch_steps(opt, train_data)

    train_log = open(opt.train_log_path, 'w')
    msg = 'Total_step_per_E: '+str(expected_total_step_epoch)
    print(msg)
    train_log.writelines([msg+'\n'])

    train_dataloader = iter(train_dataloader)
    model.train()

    train_loss = 0
    train_acc = 0

    while True:
        try:
            batch_x, batch_y = next(train_dataloader)
        except StopIteration:
            break
        
        batch_x = batch_x.cuda(non_blocking=True)
        batch_y = batch_y.cuda(non_blocking=True)

        # cut to 200-400
        length = torch.randint(200, 400, [])
        start = torch.randint(0, 400-length, [])
        batch_x = batch_x[:, start:start+length, :]

        loss, predict, emb, acc, inter = model(batch_x, batch_y, mod='train')

        train_loss += loss.item()/opt.print_freq

        train_acc += acc/opt.print_freq

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_step += 1

        if total_step in opt.lr_decay_step:
            scheduler.step() 

        if ((count+1) % opt.print_freq) == 0:
            delta_time = time.time() - timing_point
            timing_point = time.time()

            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']

            standard_freq_logging(delta_time, total_step, train_loss, train_acc, current_lr, train_log, tbx_writer)

            train_loss = 0
            train_acc = 0
        
        
        if ((count+1) % opt.val_interval_step) == 0:
            vox1test_cls_eval(model, opt, total_step, train_log, tbx_writer)
            vox1test_ASV_eval(model, opt, total_step, train_log, tbx_writer)
            sdsvc_cls_eval(model, opt, total_step, train_log, tbx_writer)
            sdsvc_ASV_eval(model, opt, total_step, train_log, tbx_writer)

            vox1test_metric_saver(model, opt, total_step, optimizer, scheduler, train_log)

            torch.backends.cudnn.benchmark = opt.cudnn_benchmark
            model.train()
