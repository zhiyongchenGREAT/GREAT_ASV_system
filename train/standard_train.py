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

    if torch.cuda.is_available():
        print("Data Parallel on ", torch.cuda.device_count(), "GPUs!")
    else:
        print("No GPU available!")
        sys.exit(1)


    if opt.model_load_path != '':
        checkpoint = torch.load(opt.model_load_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)

    model = model.to(device)
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

    # tensorboard writer
    writer_path = os.path.join(opt.log_path, 'tbx_log', opt.train_name)
    writer = SummaryWriter(log_dir=writer_path)
    if opt.tbx_path != '':
        writer_aux_path = os.path.join(opt.tbx_path, opt.train_name)
        writer_aux = SummaryWriter(log_dir=writer_aux_path)

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

    for epoch in range(start_epoch, opt.max_epoch):

        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        model.train()

        for count, (batch_x, batch_y) in enumerate(train_dataloader):

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
