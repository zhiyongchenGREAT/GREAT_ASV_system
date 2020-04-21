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
from models import *

def main():
    opt = config.Config()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    torch.backends.cudnn.benchmark = opt.cudnn_benchmark

    print('GPU ID:', opt.gpu_id)
    [print('CUDA_device:', torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model_lab.DANN_tester_AL_w_changeadv(opt.model_settings)

    if torch.cuda.is_available():
        print("Data Parallel on ", torch.cuda.device_count(), "GPUs!")
    else:
        print("No GPU available!")
        sys.exit(1)

    
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    opt_e, opt_c, opt_d = model.get_optimizer()

    scheduler_e = torch.optim.lr_scheduler.StepLR(opt_e, step_size=1, gamma=0.1)
    scheduler_c = torch.optim.lr_scheduler.StepLR(opt_c, step_size=1, gamma=0.1)
    scheduler_d = torch.optim.lr_scheduler.StepLR(opt_d, step_size=1, gamma=0.1)

    train_data = PickleDataSet_single(opt.train_list)
    train_dataloader = My_DataLoader(train_data, batch_size=opt.train_batch_size, shuffle=False, \
    sampler=RandomSampler(train_data, replacement=True, num_samples=opt.max_step*opt.train_batch_size), \
    batch_sampler=None, num_workers=opt.num_workers, collate_fn=None, \
    pin_memory=False, drop_last=False, timeout=0, \
    worker_init_fn=None, multiprocessing_context=None)


    model, optimizer, scheduler, total_step = training_utils.resume_training(opt, model, opt_e, scheduler_e)

    opt = training_utils.dir_init(opt)

    tbx_writer = training_utils.tensorboard_init(opt)

    model = model.to(device)       

    timing_point = time.time()
    expected_total_step_epoch = training_utils.get_epoch_steps(opt, train_data)

    train_log = open(opt.train_log_path, 'w')
    msg = 'Total_step_per_E: '+str(expected_total_step_epoch)
    print(msg)
    train_log.writelines([msg+'\n'])

    train_dataloader = iter(train_dataloader)
    model.train()

    print(model)

    train_loss_c = 0.0
    train_loss_d = 0.0
    train_loss_al = 0.0
    
    train_acc_c = 0.0
    train_acc_d = 0.0

    beta_scale = 1.0
    gamma_scale = 1.0

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

        [loss_c, loss_d, loss_al], _, _, [acc_c, acc_d], _ = model(batch_x, batch_y, mod='train')

        train_loss_c += loss_c.item()/opt.print_freq
        train_loss_d += loss_d.item()/opt.print_freq
        train_loss_al += loss_al.item()/opt.print_freq

        train_acc_c += acc_c/opt.print_freq
        train_acc_d += acc_d/opt.print_freq


        opt_e.zero_grad()
        opt_c.zero_grad()
        opt_d.zero_grad()
        beta = min((total_step / 10000)*0.2, 0.2)*beta_scale

        (loss_c + beta*loss_al).backward(retain_graph=True)
        opt_c.step()
        opt_e.step()
            
        opt_e.zero_grad()
        opt_c.zero_grad()
        opt_d.zero_grad()
        gamma = min((total_step / 10000)*0.5, 0.5)*gamma_scale

        (gamma*loss_d).backward()
        opt_d.step() 

        total_step += 1

        if (total_step % opt.print_freq) == 0:
            delta_time = time.time() - timing_point
            timing_point = time.time()

            training_utils.da_freq_logging(delta_time, total_step, [train_loss_c, train_loss_d, train_loss_al], \
            [train_acc_c, train_acc_d], opt_e, train_log, tbx_writer)

            msg = "ACC_d: "+str(train_acc_d)+" Beta: "+str(beta)+" Gamma: "+str(gamma)
            print(msg)
            train_log.writelines([msg+'\n'])

            # if train_acc_d < 0.8:
            #     gamma_scale = min(gamma_scale * 1.5, 20)
            # else:
            #     gamma_scale = 1.0

            train_loss_c = 0.0
            train_loss_d = 0.0
            train_loss_al = 0.0
            
            train_acc_c = 0.0
            train_acc_d = 0.0
        
        
        if (total_step % opt.val_interval_step) == 0:
            training_utils.vox1test_cls_eval_AL(model, opt, total_step, opt_e, train_log, tbx_writer)
            training_utils.vox1test_ASV_eval(model, device, opt, total_step, opt_e, train_log, tbx_writer)
            training_utils.sdsvc_cls_eval_AL(model, opt, total_step, opt_e, train_log, tbx_writer)
            training_utils.sdsvc_ASV_eval(model, device, opt, total_step, opt_e, train_log, tbx_writer)

            training_utils.vox1test_lr_decay_ctrl_AL(opt, total_step, opt_e, [scheduler_e, scheduler_c, scheduler_d], train_log)

            training_utils.vox1test_metric_saver(model, opt, total_step, opt_e, scheduler_e, train_log)

            if training_utils.stop_ctrl_std(opt, scheduler_e): break

            torch.backends.cudnn.benchmark = opt.cudnn_benchmark
            model.train()
    
    msg = "Finish training "+opt.train_name+" with Step: "+ str(total_step)
    print(msg)
    train_log.writelines([msg+'\n'])    
    train_log.close()
    tbx_writer.close()

if __name__ == '__main__':
    main()
