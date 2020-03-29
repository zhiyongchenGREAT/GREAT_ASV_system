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
# from tensorboardX import SummaryWriter
import utils
from models.tester import *
torch.multiprocessing.set_sharing_strategy('file_system')

def making_log(log_path, continue_indicator=None):
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    # StreamHandler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level=logging.INFO)
    logger.addHandler(stream_handler)
    # FileHandler
    if continue_indicator is not None:
        file_handler = logging.FileHandler(os.path.join(log_path,'train_'+continue_indicator+'.log'))
    else:
        file_handler = logging.FileHandler(os.path.join(log_path,'train.log'))
    file_handler.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger



if __name__ == '__main__':
    opt = Config()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    torch.backends.cudnn.benchmark = opt.cudnn_benchmark

    print('GPU ID:', opt.gpu_id)
    [print('CUDA_device:', torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = get_model(opt.model, opt.metric, 0, opt.model_settings, opt)
    # model = DANN_tester_3step(opt.model_settings)
    # model = DANN_tester_AL(opt.model_settings)
    if torch.cuda.is_available():
        print("Data Parallel on ", torch.cuda.device_count(), "GPUs!")
    else:
        print("No GPU available!")
        sys.exit(1)


    if opt.model_load_path != '':
        checkpoint = torch.load(opt.model_load_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)

    model = model.to(device)

    if opt.optimizer == 'sgd':
        # optimizer = torch.optim.SGD(
        #     [{"params": model.module.module.backbone.parameters(), "lr": opt.lr}, {"params": model.module.module.metrics.parameters(), "lr": opt.lr*100}], 
        #     lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay
        # )
        # optimizer = torch.optim.SGD(
        #     [{"params": model.backbone.parameters(), "lr": opt.lr}, {"params": model.metrics.parameters(), "lr": opt.lr*0.5}], 
        #     lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay
        # )    
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
        # opt_e, opt_c, opt_d = model.get_optimizer()
    else:
        print('Invalid optimizer')
        sys.exit(1)
    
    scheduler = MultiStepLR(optimizer, milestones=opt.lr_milestones, gamma=opt.lr_decay)
    # scheduler_e = MultiStepLR(opt_e, milestones=opt.lr_milestones, gamma=opt.lr_decay)
    # scheduler_c = MultiStepLR(opt_c, milestones=opt.lr_milestones, gamma=opt.lr_decay)
    # scheduler_d = MultiStepLR(opt_d, milestones=opt.lr_milestones, gamma=opt.lr_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2, eta_min=1e-4, last_epoch=-1)

    # train_data = PickleDataSet(opt.train_list)
    # train_dataloader = My_DataLoader(train_data, batch_size=None, shuffle=False, sampler=RandomSampler(train_data, replacement=True),\
    # batch_sampler=None, num_workers=opt.num_workers, collate_fn=None,\
    # pin_memory=False, drop_last=False, timeout=0,\
    # worker_init_fn=None, multiprocessing_context=None)

    train_data = PickleDataSet_single(opt.train_list)
    train_dataloader = My_DataLoader(train_data, batch_size=opt.train_batch_size, shuffle=False, sampler=RandomSampler(train_data, replacement=True),\
    batch_sampler=None, num_workers=opt.num_workers, collate_fn=None,\
    pin_memory=False, drop_last=False, timeout=0,\
    worker_init_fn=None, multiprocessing_context=None)

    val_data = PickleDataSet(opt.val_list)
    val_dataloader = My_DataLoader(val_data, batch_size=None, shuffle=False, sampler=None,\
    batch_sampler=None, num_workers=opt.num_workers, collate_fn=None,\
    pin_memory=False, drop_last=False, timeout=0,\
    worker_init_fn=None, multiprocessing_context=None)

    # initialize the logging dirs
    desc_log_path = os.path.join(opt.log_path, 'desc.log')
    val_log_path = os.path.join(opt.log_path, 'val.log')
    if opt.model_load_path == '':
        if os.path.isdir(opt.exp_top_dir):
            print('Backup the existing train exp dir to ', opt.exp_top_dir+'.bk')
            if os.path.isdir(opt.exp_top_dir+'.bk'):
                shutil.rmtree(opt.exp_top_dir+'.bk')
            shutil.move(opt.exp_top_dir, opt.exp_top_dir+'.bk')
        
        if not os.path.isdir(opt.log_path):
            os.makedirs(opt.log_path)
        if not os.path.isdir(opt.checkpoints_path):
            os.makedirs(opt.checkpoints_path)
        if not os.path.isdir(opt.train_emb_plot_path):
            os.makedirs(opt.train_emb_plot_path)
        if not os.path.isdir(opt.test_emb_plot_path):
            os.makedirs(opt.test_emb_plot_path)
        if not os.path.isdir(opt.trial_plot_path):
            os.makedirs(opt.trial_plot_path)

        shutil.copy2(opt.config_path, os.path.join(opt.exp_top_dir, 'config.log'))    

        with open(desc_log_path, 'w') as f:
            f.write(opt.description)       
        with open(val_log_path, 'a') as f:
            f.write('Val log for '+opt.train_name+'\n')
    else:
        shutil.copy2(opt.config_path, os.path.join(opt.exp_top_dir, 'config_'+opt.continue_indicator+'.log'))

    # main logger
    logger = making_log(opt.log_path, opt.continue_indicator)
    logger.info('Train Name:'+opt.train_name)
    logger.info('Description:'+opt.description)
    logger.info(model)

    total_step = 0
    start_epoch = 0
    min_eer = 1.0
    expected_total_step = len(train_data) * opt.max_epoch // opt.train_batch_size
    expected_total_step_epoch = len(train_data) // opt.train_batch_size
    logger.info('Total_step: '+str(expected_total_step))
    # model.set_totalstep(expected_total_step)

    if opt.model_load_path != '':
        print("Restoring from "+opt.model_load_path)
        total_step = checkpoint['total_step']
        start_epoch = checkpoint['start_epoch']
        min_eer = checkpoint['min_eer']
        optimizer.load_state_dict(checkpoint['optimizer'])

        sch_statdict = scheduler.state_dict()
        sch_statdict['last_epoch'] = checkpoint['scheduler']['last_epoch'] - 1
        sch_statdict['_step_count'] = checkpoint['scheduler']['_step_count'] - 1
        logger.info(sch_statdict)
        scheduler.load_state_dict(sch_statdict)
        scheduler.step()        

    timing_point = time.time()
    gamma_scale = 1.0
    beta_scale = 1.0

    for epoch in range(start_epoch, opt.max_epoch):

        train_loss = 0
        train_acc = 0
        train_acc_d = 0
        val_loss = 0
        val_acc = 0
        val_acc_d = 0
        train_acc_d_l50 = 0.5

        model.train()

        for count, (batch_x, batch_y) in enumerate(train_dataloader):

            batch_x = batch_x.cuda(non_blocking=True)
            batch_y = batch_y.cuda(non_blocking=True)
            # batch_y = batch_y - 1211
            # print(batch_y)

            # cut to 200-400
            length = torch.randint(200, 400, [])
            start = torch.randint(0, 400-length, [])
            batch_x = batch_x[:, start:start+length, :]

            # [loss_c, loss_d, loss_al], predict, emb, acc, acc_d = model(batch_x, batch_y, mod='train')
            loss, predict, emb, acc, acc_d = model(batch_x, batch_y, mod='train')

            # train_loss += loss_c.item()
            train_loss += loss.item()

            train_acc += acc

            train_acc_d += acc_d

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if ((count+1) % 5) != 0:
            #     # print('e')
            #     # opt_e.zero_grad()
            #     # opt_c.zero_grad()
            #     # opt_d.zero_grad()
            #     # loss_c.backward()
            #     # opt_c.step()

            #     opt_e.zero_grad()
            #     opt_c.zero_grad()
            #     opt_d.zero_grad()
            #     beta = min(total_step / 20000, 1.0)
            #     # beta = 2. / (1. + np.exp(-10 * p)) - 1
            #     # if beta == 1.0:
            #     beta = beta*beta_scale

            #     (loss_c + beta*loss_al).backward()
            #     opt_c.step()
            #     opt_e.step()
                
            # else:
                
            #     opt_e.zero_grad()
            #     opt_c.zero_grad()
            #     opt_d.zero_grad()

            #     gamma = min(total_step / 20000, 1.0)
            #     gamma = gamma*gamma_scale
            #     (gamma*loss_d).backward()
            #     opt_d.step()           

            total_step += 1

            if total_step in opt.lr_decay_step:
                scheduler_e.step() 
                scheduler_c.step() 
                scheduler_d.step() 

            if ((count+1) % opt.print_freq) == 0:
                # print('beta:', beta)
                # print('gamma:', gamma)
                logger.info('----------')
                delta_time = time.time() - timing_point
                logger.info('Dur:'+str(delta_time))
                timing_point = time.time()
                train_loss = train_loss/opt.print_freq
                train_acc = train_acc/opt.print_freq
                train_acc_d = train_acc_d/opt.print_freq
                train_acc_d_l50 = train_acc_d

                # if train_acc_d_l50 > 0.75:
                #     gamma_scale = 0.75*gamma_scale
                # else:
                #     gamma_scale = min((1/0.75)*gamma_scale, 1.0)


                # if train_acc_d_l50 < 0.4:
                #     beta_scale = 0.75*beta_scale
                # else:
                #     beta_scale = min((1/0.75)*beta_scale, 1.0)
                if train_acc_d_l50 > 0.8:
                    # gamma_scale = 0.75*gamma_scale
                    beta_scale = min((1/0.75)*beta_scale, 10.0)
                elif train_acc_d_l50 < 0.4:
                    beta_scale = 0.75*beta_scale
                else:
                    # gamma_scale = min((1/0.75)*gamma_scale, 1.0)
                    beta_scale = 1.0



                # for param_group in opt_e.param_groups:
                #     current_lr = param_group['lr']
                for param_group in optimizer.param_groups:
                    current_lr = param_group['lr']

                logger.info('Pro: '+'{0:.3f}'.format(total_step * (1.0 / expected_total_step_epoch))+'/'+str(opt.max_epoch)+' Epoch:'+str(epoch+1)+' Step:'+str(count+1)+' Loss:'+str(train_loss)+' Accd:'+str(train_acc_d )+' Acc:'+str(train_acc)+' lr:'+str(current_lr))
 
 
                train_loss = 0
                train_acc = 0
                train_acc_d = 0
            
            
            if opt.val_interval_step is not None and ((count+1) % opt.val_interval_step) == 0:
                pass
                torch.backends.cudnn.benchmark = opt.cudnn_benchmark
                model.train()
                
        # # Eval model at the end of each epoch
        model.eval()
        for val_count, (val_x, val_y) in enumerate(val_dataloader):

            val_x = val_x.cuda(non_blocking=True)
            val_y = val_y.cuda(non_blocking=True)

            with torch.no_grad():
                # [loss, _, _], predict, emb, acc, acc_d = model(val_x, val_y, mod='eval')
                loss, predict, emb, acc, acc_d = model(val_x, val_y, mod='eval')
            val_loss += loss.item()
            val_acc += acc
            val_acc_d += acc_d

        val_loss = val_loss / (val_count + 1)  
        val_acc = val_acc / (val_count + 1)
        val_acc_d = val_acc_d / (val_count + 1)

        if (epoch+1) in opt.resulting_epochs:
            temporal_results_path = os.path.join(opt.temporal_results_path, 'e'+str(epoch+1)+'s'+str(count+1)+'end')
            if not os.path.isdir(temporal_results_path):
                os.makedirs(temporal_results_path)
            eer, minc, actc = utils.trial_eval(model, opt, device, temporal_results_path)
            eer_2, minc_2, actc_2 = utils.trial_eval_2(model, opt, device, temporal_results_path)

            with open(os.path.join(temporal_results_path, 'eer'), 'w') as f:
                f.write('Result eer:'+str(eer*100)+'%')

            # if eer < min_eer:
            #     min_eer = eer
            #     state = {
            #         'model': model.state_dict(),
            #         'optimizer': optimizer.state_dict(),
            #         'scheduler': scheduler.state_dict(),
            #         'start_epoch': epoch+1,
            #         'total_step': total_step,
            #         'min_eer': min_eer
            #         }
            #     model_name = os.path.join(opt.checkpoints_path, 'min_eer.model')
            #     torch.save(state, model_name)
            #     logger.info('----------')
            #     logger.info('Model saved to '+model_name)
            #     with open(os.path.join(opt.checkpoints_path, 'min_eer.log'), 'w') as f:
            #         f.write('e'+str(epoch+1)+'s'+str(count+1)+'end'+'\n')
        else:
            eer = None

        logger.info('----------')
        logger.info('End Epoch:'+str(epoch+1)+' ValLoss:'+str(val_loss)+' ValAcc:'+str(val_acc)+' EER:'+str(eer)+' minc:'+str(minc)+' actc:'+str(actc))
        with open(val_log_path, 'a') as f:
            f.write('1ValLoss:{:.5f} ValAcc:{:.5f} ValAccD:{:.5f} EER:{:.5f} minc:{:.5f} actc:{:.5f}\n'.format(val_loss, val_acc, val_acc_d, eer, minc, actc))
            f.write('2ValLoss:{:.5f} ValAcc:{:.5f} ValAccD:{:.5f} EER:{:.5f} minc:{:.5f} actc:{:.5f}\n'.format(val_loss, val_acc, val_acc_d, eer_2, minc_2, actc_2))                   

        
        val_loss = 0
        val_acc = 0
        val_acc_d = 0

        if opt.lr_decay_step == []:
            # scheduler_e.step()
            # scheduler_c.step() 
            # scheduler_d.step() 
            scheduler.step()

        # save model for epochs
        # if (epoch+1) in opt.ckpt_epochs: 
        #     state = {
        #         'model': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'scheduler': scheduler.state_dict(),
        #         'start_epoch': epoch+1,
        #         'total_step': total_step,
        #         'min_eer': min_eer
        #         }
            
        #     model_name = os.path.join(opt.checkpoints_path, 'e'+str(epoch+1)+'s'+str(count+1)+'end'+'.model')
        #     torch.save(state, model_name)
        #     logger.info('----------')
        #     logger.info('Model saved to '+model_name)

        torch.backends.cudnn.benchmark = opt.cudnn_benchmark
        model.train()
    
    
    logger.info('Finished training '+opt.train_name)

