import os
import torch

def std_saver(model, opt, total_step, optimizer, scheduler, train_log, save_name, save_log):
    save_path = os.path.join(opt.checkpoints_path, save_name+'.model')
    
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'total_step': total_step
        }

    torch.save(state, save_path)
    msg = 'Model saved to '+save_path
    print(msg)
    train_log.writelines([msg+'\n']) 
    with open(os.path.join(opt.checkpoints_path, save_name+'.log'), 'w') as f:
        f.write(save_log)


def vox1test_metric_saver(model, opt, total_step, optimizer, scheduler, train_log):
    metric = opt.saver_metric
    metric_value = None
    save_path = os.path.join(opt.checkpoints_path, 'vox1test_metric_saver_'+metric+'.model')

    with open(opt.val_log_path, 'r') as f:
        val_list = f.readlines()
    
    for i in range(-1, -len(val_list), -1):
        if val_list[i][:-1].split(' ')[0] == 'vox1test_ASV_eval':
            break

    line_split = val_list[i][:-1].split(' ')
    for j in range(len(line_split)):
        if line_split[j] == metric+':':
            metric_value = float(line_split[j+1])

    if metric_value is None:
        raise NotImplementedError
    
    if os.path.isfile(save_path):
        with open(os.path.join(opt.checkpoints_path, 'vox1test_metric_saver.log'), 'r') as f:
            line = f.readlines()[0][:-1]
            best_value = float(line.split(' ')[-1])
            if metric_value > best_value:
                return

    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'total_step': total_step
        }

    torch.save(state, save_path)
    msg = 'Model saved to '+save_path
    print(msg)
    train_log.writelines([msg+'\n']) 
    with open(os.path.join(opt.checkpoints_path, 'vox1test_metric_saver.log'), 'w') as f:
        f.write('s'+str(total_step)+' '+metric+' '+str(metric_value))

def sdsvc_metric_saver(model, opt, total_step, optimizer, scheduler, train_log):
    metric = opt.saver_metric
    metric_value = None
    save_path = os.path.join(opt.checkpoints_path, 'sdsvc_metric_saver_'+metric+'.model')

    with open(opt.val_log_path, 'r') as f:
        val_list = f.readlines()
    
    for i in range(-1, -len(val_list), -1):
        if val_list[i][:-1].split(' ')[0] == 'sdsvc_ASV_eval':
            break

    line_split = val_list[i][:-1].split(' ')
    for j in range(len(line_split)):
        if line_split[j] == metric+':':
            metric_value = float(line_split[j+1])

    if metric_value is None:
        raise NotImplementedError
    
    if os.path.isfile(save_path):
        with open(os.path.join(opt.checkpoints_path, 'sdsvc_metric_saver.log'), 'r') as f:
            line = f.readlines()[0][:-1]
            best_value = float(line.split(' ')[-1])
            if metric_value > best_value:
                return

    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'total_step': total_step
        }

    torch.save(state, save_path)
    msg = 'Model saved to '+save_path
    print(msg)
    train_log.writelines([msg+'\n']) 
    with open(os.path.join(opt.checkpoints_path, 'sdsvc_metric_saver.log'), 'w') as f:
        f.write('s'+str(total_step)+' '+metric+' '+str(metric_value))
