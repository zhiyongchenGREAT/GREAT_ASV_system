import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../')

import torch
import numpy as np
from read_data import *
from my_dataloader import *
import score

def libri_cls_eval(model, opt, total_step, optimizer, train_log, tbx_writer):
    if not hasattr(opt, 'libri_val_list'):
        return

    val_data = PickleDataSet(opt.libri_val_list)
    val_dataloader = My_DataLoader(val_data, batch_size=None, shuffle=False, sampler=None,\
    batch_sampler=None, num_workers=opt.num_workers, collate_fn=None,\
    pin_memory=False, drop_last=False, timeout=0,\
    worker_init_fn=None, multiprocessing_context=None)   
    
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    for val_count, (val_x, val_y) in enumerate(val_dataloader):

        val_x = val_x.cuda(non_blocking=True)
        val_y = val_y.cuda(non_blocking=True)
        
        with torch.no_grad():
            loss, predict, emb, acc, _ = model(val_x, val_y, mod='eval')
        
        val_loss += loss.item()
        val_acc += acc                     

    val_loss = val_loss / (val_count + 1)  
    val_acc = val_acc / (val_count + 1)

    tbx_writer.add_scalar('libri_cls_eval_loss', val_loss, total_step)
    tbx_writer.add_scalar('libri_cls_eval_acc', val_acc, total_step)

    current_lr = optimizer.param_groups[0]['lr']

    msg = "libri_cls_eval Step: {:} Valcount: {:} ValLoss: {:.4f} ValAcc: {:.4f} Lr: {:.5f}"\
    .format(total_step, (val_count + 1), val_loss, val_acc, current_lr)
    print(msg)
    train_log.writelines([msg+'\n']) 
    with open(opt.val_log_path, 'a') as f:
        f.writelines([msg+'\n'])

def libri_cls_eval_AL(model, opt, total_step, optimizer, train_log, tbx_writer):
    if not hasattr(opt, 'libri_val_list'):
        return

    val_data = PickleDataSet(opt.libri_val_list)
    val_dataloader = My_DataLoader(val_data, batch_size=None, shuffle=False, sampler=None,\
    batch_sampler=None, num_workers=opt.num_workers, collate_fn=None,\
    pin_memory=False, drop_last=False, timeout=0,\
    worker_init_fn=None, multiprocessing_context=None)   
    
    model.eval()
    val_loss_c = 0.0
    val_loss_d = 0.0
    val_loss_al = 0.0

    val_acc_c = 0.0
    val_acc_d = 0.0 
    for val_count, (val_x, val_y) in enumerate(val_dataloader):

        val_x = val_x.cuda(non_blocking=True)
        val_y = val_y.cuda(non_blocking=True)

        with torch.no_grad():
            [loss_c, loss_d, loss_al], _, _, [acc_c, acc_d], _ = model(val_x, val_y, mod='eval')
        
        val_loss_c += loss_c.item()
        val_loss_d += loss_d.item()
        val_loss_al += loss_al.item()

        val_acc_c += acc_c
        val_acc_d += acc_d   

    val_loss_c = val_loss_c / (val_count + 1)  
    val_loss_d = val_loss_d / (val_count + 1)
    val_loss_al = val_loss_al / (val_count + 1)

    val_acc_c = val_acc_c / (val_count + 1)
    val_acc_d = val_acc_d / (val_count + 1)

    tbx_writer.add_scalar('libri_cls_eval_loss', val_loss_c, total_step)
    tbx_writer.add_scalar('libri_cls_eval_loss_d', val_loss_d, total_step)
    tbx_writer.add_scalar('libri_cls_eval_loss_al', val_loss_al, total_step)

    tbx_writer.add_scalar('libri_cls_eval_acc', val_acc_c, total_step)
    tbx_writer.add_scalar('libri_cls_eval_acc_d', val_acc_d, total_step)                 

    current_lr = optimizer.param_groups[0]['lr']

    msg = "libri_cls_eval Step: {:} Valcount: {:} ValLoss: {:.4f} ValAcc: {:.4f} ValAcc_d: {:.4f} Lr: {:.5f}"\
    .format(total_step, (val_count + 1), val_loss_c, val_acc_c, val_acc_d, current_lr)
    print(msg)
    train_log.writelines([msg+'\n']) 
    with open(opt.val_log_path, 'a') as f:
        f.writelines([msg+'\n']) 


def libri_ASV_eval(model, device, opt, total_step, optimizer, train_log, tbx_writer):
    torch.backends.cudnn.benchmark = False
    model.eval()
    # print('Final score evaluation')

    train_data = PickleDataSet(opt.libri_trial_list)
    train_dataloader = My_DataLoader(train_data, batch_size=None, shuffle=False, sampler=None,\
    batch_sampler=None, num_workers=opt.num_workers, collate_fn=None,\
    pin_memory=False, drop_last=False, timeout=0,\
    worker_init_fn=None, multiprocessing_context=None)

    test_list = {}

    for count, (batch_x, batch_y) in enumerate(train_dataloader):
        batch_x = batch_x.to(device)
        label = batch_y[0]

        batch_y = torch.tensor([0]).to(device)
        
        with torch.no_grad():
            _, _, emb, _, _ = model(batch_x, batch_y, mod='eval')

        emb = emb.squeeze().data.cpu().numpy()
        
        if label not in test_list.keys():
            test_list[label] = emb[None, :]
        else:
            print('repeat eer:', label)
            break        

    msg = "libri_ASV_eval Step: {:} Embcount: {:}".format(total_step, (count + 1))
    print(msg)
    train_log.writelines([msg+'\n']) 
    
    out_dir = os.path.join(opt.temporal_results_path, 's'+str(total_step), 'libri_ASV_eval')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    f_out = open(os.path.join(out_dir, 'scores'), 'w')   
    
    for i in test_list:
        test_list[i] = (1.0 / np.linalg.norm(test_list[i])) * test_list[i]
    
    with open(opt.libri_trial_keys, 'r') as f:
        for count, line in enumerate(f):
            if count == 0:
                pass
                # print(line)

            enroll_emb = test_list[line.split(' ')[0][:-4]].squeeze()
            test_emb = test_list[line.split(' ')[1][:-4]].squeeze()
            
            cosine = np.dot(enroll_emb, test_emb)
            
            f_out.write(line.split(' ')[0]+' '+line.split(' ')[1]+' '+str(cosine)+'\n')
    
    f_out.close()

    msg = "libri_trial_keys Step: {:} Trialcount: {:}".format(total_step, (count + 1))
    print(msg)
    train_log.writelines([msg+'\n']) 

    if hasattr(opt, 'libri_aux_list') and hasattr(opt, 'libri_aux_keys'):
        train_data = PickleDataSet(opt.libri_aux_list)
        train_dataloader = My_DataLoader(train_data, batch_size=None, shuffle=False, sampler=None,\
        batch_sampler=None, num_workers=opt.num_workers, collate_fn=None,\
        pin_memory=False, drop_last=False, timeout=0,\
        worker_init_fn=None, multiprocessing_context=None)

        test_list = {}

        for count, (batch_x, batch_y) in enumerate(train_dataloader):
            batch_x = batch_x.to(device)
            label = batch_y[0]

            batch_y = torch.tensor([0]).to(device)
            
            with torch.no_grad():
                _, _, emb, _, _ = model(batch_x, batch_y, mod='eval')

            emb = emb.squeeze().data.cpu().numpy()
            
            if label not in test_list.keys():
                test_list[label] = emb[None, :]
            else:
                print('repeat eer:', label)
                break        

        msg = "libri_ASV_eval Step: {:} AuxEmbcount: {:}".format(total_step, (count + 1))
        print(msg)
        train_log.writelines([msg+'\n']) 
        
        out_dir = os.path.join(opt.temporal_results_path, 's'+str(total_step), 'libri_ASV_eval')
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        f_out = open(os.path.join(out_dir, 'scores_aux'), 'w')   
        
        for i in test_list:
            test_list[i] = (1.0 / np.linalg.norm(test_list[i])) * test_list[i]
        
        with open(opt.libri_aux_keys, 'r') as f:
            for count, line in enumerate(f):
                if count == 0:
                    pass
                    # print(line)

                enroll_emb = test_list[line.split(' ')[0][:-4]].squeeze()
                test_emb = test_list[line.split(' ')[1][:-4]].squeeze()
                
                cosine = np.dot(enroll_emb, test_emb)
                
                f_out.write(line.split(' ')[0]+' '+line.split(' ')[1]+' '+str(cosine)+'\n')
        
        f_out.close()

        msg = "libri_ASV_eval Step: {:} TrialAuxcount: {:}".format(total_step, (count + 1))
        print(msg)
        train_log.writelines([msg+'\n'])

    if hasattr(opt, 'libri_aux_list') and hasattr(opt, 'libri_aux_keys'):
        score.calibrating(os.path.join(out_dir, 'calib.pth'), 50, opt.libri_aux_keys, [os.path.join(out_dir, 'scores_aux')])
        score.applying(os.path.join(out_dir, 'calib.pth'), [os.path.join(out_dir, 'scores')], os.path.join(out_dir, 'scores_calib'))
        eer, minc, actc = score.scoring(os.path.join(out_dir, 'scores_calib'), opt.libri_trial_keys, opt.scoring_config)
    else:
        eer, minc, actc = score.scoring(os.path.join(out_dir, 'scores'), opt.libri_trial_keys, opt.scoring_config)

    tbx_writer.add_scalar('libri_ASV_eval_EER', eer, total_step)
    tbx_writer.add_scalar('libri_ASV_eval_MINC', minc, total_step)
    tbx_writer.add_scalar('libri_ASV_eval_ACTC', actc, total_step)

    current_lr = optimizer.param_groups[0]['lr']
    msg = "libri_ASV_eval Step: {:} EER: {:.4f} MINC: {:.4f} ACTC: {:.4f} Lr: {:.5f}"\
    .format(total_step, eer, minc, actc, current_lr)
    print(msg)
    train_log.writelines([msg+'\n']) 
    with open(opt.val_log_path, 'a') as f:
        f.writelines([msg+'\n'])    
        