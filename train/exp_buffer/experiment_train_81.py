import time
import importlib

import torch
import training_utils
from torch.utils.data import *
from my_dataloader import *
from read_data import *
import config.config as config
importlib.reload(config)
from model_bank import *
import random
import numpy as np

def main():
    opt = config.Config()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    torch.backends.cudnn.benchmark = opt.cudnn_benchmark

    print('GPU ID:', opt.gpu_id)
    [print('CUDA_device:', torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = get_model(opt.model, opt.metric, opt.model_settings, opt)

    if torch.cuda.is_available():
        print("Data Parallel on ", torch.cuda.device_count(), "GPUs!")
    else:
        print("No GPU available!")
        sys.exit(1)

    
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)    
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2, eta_min=1e-4, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    train_data_s = PickleDataSet_single(opt.train_list)
    train_dataloader_s = My_DataLoader(train_data_s, batch_size=opt.train_batch_size//2, shuffle=False, \
    sampler=RandomSampler(train_data_s, replacement=True, num_samples=opt.max_step*opt.train_batch_size//2), \
    batch_sampler=None, num_workers=opt.num_workers, collate_fn=None, \
    pin_memory=False, drop_last=False, timeout=0, \
    worker_init_fn=None, multiprocessing_context=None)

    train_data_t = PickleDataSet_single(opt.train_list)
    train_dataloader_t = My_DataLoader(train_data_t, batch_size=opt.train_batch_size//2, shuffle=False, \
    sampler=RandomSampler(train_data_t, replacement=True, num_samples=opt.max_step*opt.train_batch_size//2), \
    batch_sampler=None, num_workers=opt.num_workers, collate_fn=None, \
    pin_memory=False, drop_last=False, timeout=0, \
    worker_init_fn=None, multiprocessing_context=None)

    model, optimizer, scheduler, total_step = training_utils.resume_training(opt, model, optimizer, scheduler)

    opt = training_utils.dir_init(opt)

    tbx_writer = training_utils.tensorboard_init(opt)

    model = model.to(device)       

    timing_point = time.time()
    expected_total_step_epoch = training_utils.get_epoch_steps(opt, train_data_s)

    train_log = open(opt.train_log_path, 'w')
    msg = 'Total_step_per_E: '+str(expected_total_step_epoch)
    print(msg)
    train_log.writelines([msg+'\n'])

    train_dataloader_s = iter(train_dataloader_s)
    train_dataloader_t = iter(train_dataloader_t)
    model.train()

    print(model)

    train_loss = 0
    train_acc = 0

    while True:
        try:
            batch_x_s, batch_y_s = next(train_dataloader_s)
            batch_x_t, batch_y_t = next(train_dataloader_t)

            batch_x = torch.cat([batch_x_s, batch_x_t], axis=0)
            batch_y = torch.cat([batch_y_s, batch_y_t], axis=0)
            index_shuffle = np.arange(0, batch_x.size(0))
            random.shuffle(index_shuffle)
            batch_x = batch_x[index_shuffle]
            batch_y = batch_y[index_shuffle]
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

        if (total_step % opt.print_freq) == 0:
            delta_time = time.time() - timing_point
            timing_point = time.time()

            training_utils.standard_freq_logging(delta_time, total_step, train_loss, train_acc, optimizer, train_log, tbx_writer)

            train_loss = 0
            train_acc = 0
        
        
        if (total_step % opt.val_interval_step) == 0:
            training_utils.vox1test_cls_eval(model, opt, total_step, optimizer, train_log, tbx_writer)
            training_utils.vox1test_ASV_eval(model, device, opt, total_step, optimizer, train_log, tbx_writer)
            training_utils.sdsvc_cls_eval(model, opt, total_step, optimizer, train_log, tbx_writer)
            training_utils.sdsvc_ASV_eval(model, device, opt, total_step, optimizer, train_log, tbx_writer)

            training_utils.vox1test_lr_decay_ctrl(opt, total_step, optimizer, scheduler, train_log)

            training_utils.vox1test_metric_saver(model, opt, total_step, optimizer, scheduler, train_log)

            if training_utils.stop_ctrl_std(opt, scheduler): break

            torch.backends.cudnn.benchmark = opt.cudnn_benchmark
            model.train()
    
    msg = "Finish training "+opt.train_name+" with Step: "+ str(total_step)
    print(msg)
    train_log.writelines([msg+'\n'])    
    train_log.close()
    tbx_writer.close()

if __name__ == '__main__':
    main()
