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

    val_data = PickleDataSet(opt.val_list)
    val_dataloader = My_DataLoader(val_data, batch_size=None, shuffle=False, sampler=None,\
    batch_sampler=None, num_workers=opt.num_workers, collate_fn=None,\
    pin_memory=False, drop_last=False, timeout=0,\
    worker_init_fn=None, multiprocessing_context=None)


    # total_step = 0
    # start_epoch = 0

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


    train_loss = 0
    train_acc = 0
    val_loss = 0
    val_acc = 0

    train_dataloader = iter(train_dataloader)
    model.train()

    for count, (batch_x, batch_y) in enumerate(train_dataloader):

        try:
            batch_x, batch_y = next(train_dataloader)
        
        batch_x = batch_x.cuda(non_blocking=True)
        batch_y = batch_y.cuda(non_blocking=True)

        # cut to 200-400
        length = torch.randint(200, 400, [])
        start = torch.randint(0, 400-length, [])
        batch_x = batch_x[:, start:start+length, :]

        loss, predict, emb, acc, inter = model(batch_x, batch_y, mod='train')

        train_loss += loss.item()

        train_acc += acc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_step += 1

        if total_step in opt.lr_decay_step:
            scheduler.step() 

        if ((count+1) % opt.print_freq) == 0:
            logger.info('----------')
            delta_time = time.time() - timing_point
            logger.info('Dur:'+str(delta_time))
            timing_point = time.time()
            train_loss = train_loss/opt.print_freq
            train_acc = train_acc/opt.print_freq

            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']

            logger.info('Pro: '+'{0:.3f}'.format(total_step * (1.0 / expected_total_step_epoch))+'/'+str(opt.max_epoch)+' Epoch:'+str(epoch+1)+' Step:'+str(count+1)+' Loss:'+str(train_loss)+' Inter:'+str(inter)+' Acc:'+str(train_acc)+' lr:'+str(current_lr))

            writer.add_scalar('train/loss', train_loss, total_step)
            writer.add_scalar('train/acc', train_acc, total_step)
            writer.add_scalar('train/lr', current_lr, total_step)
            writer.add_scalar('train/inter_power', inter, total_step)              
            writer_aux.add_scalar('train/loss', train_loss, total_step)
            writer_aux.add_scalar('train/acc', train_acc, total_step)
            writer_aux.add_scalar('train/lr', current_lr, total_step)
            writer_aux.add_scalar('train/inter_power', inter, total_step)  
            train_loss = 0
            train_acc = 0
        
        
        if opt.val_interval_step is not None and ((count+1) % opt.val_interval_step) == 0:
            model.eval()
            for val_count, (val_x, val_y) in enumerate(val_dataloader):

                val_x = val_x.cuda(non_blocking=True)
                val_y = val_y.cuda(non_blocking=True)
                
                with torch.no_grad():
                    loss, predict, emb, acc, _ = model(val_x, val_y, mod='eval')
                val_loss += loss.item()

                val_acc += acc                     

            val_loss = val_loss / (val_count + 1)  
            val_acc = val_acc / (val_count + 1)

            writer.add_scalar('val/loss', val_loss, total_step)
            writer.add_scalar('val/acc', val_acc, total_step)
            writer_aux.add_scalar('val/loss', val_loss, total_step)
            writer_aux.add_scalar('val/acc', val_acc, total_step)

            if (epoch+1) in opt.fine_collecting:
                temporal_results_path = os.path.join(opt.temporal_results_path, 'e'+str(epoch+1)+'s'+str(count+1))
                if not os.path.isdir(temporal_results_path):
                    os.makedirs(temporal_results_path)
                eer, minc, actc = utils.trial_eval(model, opt, device, temporal_results_path)
                # fig.savefig(os.path.join(temporal_results_path, 'result.png'), dpi=100)
                # fig.savefig(os.path.join(opt.trial_plot_path, 'e'+str(epoch+1)+'s'+str(count+1)+'.png'), dpi=100)
                # writer.add_figure('trial_plot', fig, total_step)                                
                writer.add_scalar('val/eer', eer, total_step)
                # writer_aux.add_figure('trial_plot', fig, total_step)                                
                writer_aux.add_scalar('val/eer', eer, total_step)
                with open(os.path.join(temporal_results_path, 'eer'), 'w') as f:
                    f.write('Result eer:'+str(eer*100)+'%')

                if opt.saveall:
                    state = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'start_epoch': epoch+1,
                        'total_step': total_step,
                        'min_eer': min_eer
                        }
                    model_name = os.path.join(opt.checkpoints_path, 'e'+str(epoch+1)+'s'+str(count+1)+'.model')
                    torch.save(state, model_name)
                    logger.info('----------')
                    logger.info('Model saved to '+model_name)

                if eer < min_eer:
                    min_eer = eer
                    state = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'start_epoch': epoch+1,
                        'total_step': total_step,
                        'min_eer': min_eer
                        }
                    model_name = os.path.join(opt.checkpoints_path, 'min_eer.model')
                    torch.save(state, model_name)
                    logger.info('----------')
                    logger.info('Model saved to '+model_name)
                    with open(os.path.join(opt.checkpoints_path, 'min_eer.log'), 'w') as f:
                        f.write('e'+str(epoch+1)+'s'+str(count+1)+'\n')
            else:
                eer = None

            logger.info('----------')
            logger.info('ValLoss:'+str(val_loss)+' ValAcc:'+str(val_acc)+' EER:'+str(eer)+' minc:'+str(minc)+' actc:'+str(actc))
            with open(val_log_path, 'a') as f:
                f.write('ValLoss:'+str(val_loss)+' ValAcc:'+str(val_acc)+' EER:'+str(eer)+' minc:'+str(minc)+' actc:'+str(actc)+'\n')                    
            val_loss = 0
            val_acc = 0  
            torch.backends.cudnn.benchmark = opt.cudnn_benchmark
            model.train()
                
    # # Eval model at the end of each epoch
    model.eval()
    for val_count, (val_x, val_y) in enumerate(val_dataloader):

        val_x = val_x.cuda(non_blocking=True)
        val_y = val_y.cuda(non_blocking=True)

        with torch.no_grad():
            loss, predict, emb, acc, _ = model(val_x, val_y, mod='eval')
        val_loss += loss.item()
        val_acc += acc  

    val_loss = val_loss / (val_count + 1)  
    val_acc = val_acc / (val_count + 1)

    writer.add_scalar('val/loss', val_loss, total_step)
    writer.add_scalar('val/acc', val_acc, total_step)
    writer_aux.add_scalar('val/loss', val_loss, total_step)
    writer_aux.add_scalar('val/acc', val_acc, total_step)

    if (epoch+1) in opt.resulting_epochs:
        temporal_results_path = os.path.join(opt.temporal_results_path, 'e'+str(epoch+1)+'s'+str(count+1)+'end')
        if not os.path.isdir(temporal_results_path):
            os.makedirs(temporal_results_path)
        eer, minc, actc = utils.trial_eval(model, opt, device, temporal_results_path)
        # fig.savefig(os.path.join(temporal_results_path, 'result.png'), dpi=100)
        # fig.savefig(os.path.join(opt.trial_plot_path, 'e'+str(epoch+1)+'s'+str(count+1)+'end.png'), dpi=100)
        # writer.add_figure('trial_plot', fig, total_step)                                
        writer.add_scalar('val/eer', eer, total_step)
        # writer_aux.add_figure('trial_plot', fig, total_step)                                
        writer_aux.add_scalar('val/eer', eer, total_step)
        with open(os.path.join(temporal_results_path, 'eer'), 'w') as f:
            f.write('Result eer:'+str(eer*100)+'%')

        if eer < min_eer:
            min_eer = eer
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'start_epoch': epoch+1,
                'total_step': total_step,
                'min_eer': min_eer
                }
            model_name = os.path.join(opt.checkpoints_path, 'min_eer.model')
            torch.save(state, model_name)
            logger.info('----------')
            logger.info('Model saved to '+model_name)
            with open(os.path.join(opt.checkpoints_path, 'min_eer.log'), 'w') as f:
                f.write('e'+str(epoch+1)+'s'+str(count+1)+'end'+'\n')
    else:
        eer = None

    logger.info('----------')
    logger.info('End Epoch:'+str(epoch+1)+' ValLoss:'+str(val_loss)+' ValAcc:'+str(val_acc)+' EER:'+str(eer)+' minc:'+str(minc)+' actc:'+str(actc))
    with open(val_log_path, 'a') as f:
        f.write('End Epoch:'+str(epoch+1)+' ValLoss:'+str(val_loss)+' ValAcc:'+str(val_acc)+' EER:'+str(eer)+' minc:'+str(minc)+' actc:'+str(actc)+'\n')
    
    val_loss = 0
    val_acc = 0

    if opt.lr_decay_step == []:
        scheduler.step()
    # save model for epochs
    if (epoch+1) in opt.ckpt_epochs: 
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'start_epoch': epoch+1,
            'total_step': total_step,
            'min_eer': min_eer
            }
        
        model_name = os.path.join(opt.checkpoints_path, 'e'+str(epoch+1)+'s'+str(count+1)+'end'+'.model')
        torch.save(state, model_name)
        logger.info('----------')
        logger.info('Model saved to '+model_name)

    torch.backends.cudnn.benchmark = opt.cudnn_benchmark
    model.train()
    
    
    logger.info('Finished training '+opt.train_name)

    writer.close()
    writer_aux.close()
