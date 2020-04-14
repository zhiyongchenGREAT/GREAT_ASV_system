#!/usr/bin/env python

import os
import time
import sys
import argparse

import torch
from torch.utils.data import DataLoader, RandomSampler
from my_dataloader import My_DataLoader
from read_data import CSVDataSet, WithReplacementRandomSampler, PickleDataSet, PickleDataSet_single
from models.model_lab import *

def main(GPU_O, compute, mem):
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_O

    print('GPU ID:', GPU_O)
    [print('CUDA_device:', torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = GAN_tester({'in_feat': 30, 'emb_size': 512, 'class_num': 1699, 's': 50, 'm': 0.2, 'anneal_steps': 1000, 'lmd_inter': 0.1})
    if torch.cuda.is_available():
        print("Data Parallel on ", torch.cuda.device_count(), "GPUs!")
    else:
        print("No GPU available!")
        sys.exit(1)

    model = model.to(device)

    opt_e, opt_c, opt_d = model.get_optimizer()

    total_step = 0
    start_epoch = 0
    min_eer = 1.0        

    timing_point = time.time()
    gamma_scale = 1.0
    beta_scale = 1.0

    compute_sig = True

    for epoch in range(0, int(10e10)):
        print('Occupying!', GPU_O)

        train_loss = 0
        train_acc = 0
        train_acc_d = 0
        val_loss = 0
        val_acc = 0
        val_acc_d = 0
        train_acc_d_l50 = 0.5

        model.train()

        for count in range(0, int(10e10)):

            batch_x_s = torch.randn([mem, 300, 30]).cuda(non_blocking=True)
            batch_y_s = torch.randint(1699, (mem,)).cuda(non_blocking=True)
            batch_x_t = torch.randn([mem, 300, 30]).cuda(non_blocking=True)
            batch_y_t = torch.randint(1699, (mem,)).cuda(non_blocking=True)

            if compute_sig:
                # cut to 200-400
                length = torch.randint(200, 400, [])
                start = torch.randint(0, 400-length, [])
                batch_x_s = batch_x_s[:, start:start+length, :]
                batch_x_t = batch_x_t[:, start:start+length, :]

                batch_x = torch.cat([batch_x_s, batch_x_t], axis=0)
                batch_y = torch.cat([batch_y_s, batch_y_t], axis=0)

                [loss_c, loss_d, loss_al], predict, emb, acc, acc_d = model(batch_x, batch_y, mod='train')

                train_loss += loss_c.item()

                train_acc += acc

                train_acc_d += acc_d


                opt_e.zero_grad()
                opt_c.zero_grad()
                opt_d.zero_grad()

                (loss_c + loss_al + loss_d).backward()
                opt_c.step()
                opt_e.step()
                opt_d.step()
            
            if compute == 0:
                compute_sig = False                

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scorer.')
    parser.add_argument("-gpu", "--gpu", type=str, required=True)
    parser.add_argument("-compute", "--compute", type=int, required=True)
    parser.add_argument("-mem", "--mem", type=int, required=True)

    args = parser.parse_args()

    GPU_O = args.gpu
    compute = args.compute
    mem = args.mem
    main(GPU_O, compute, mem)
