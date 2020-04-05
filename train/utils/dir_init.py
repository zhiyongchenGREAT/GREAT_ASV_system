import os
import shutil

def dir_init(opt):
    exp_top_dir = os.path.join(opt.exp_path, opt.train_name)
    opt.exp_top_dir = exp_top_dir

    if os.path.isdir(opt.exp_top_dir):
        print('Backup the existing train exp dir to ', exp_top_dir+'.bk')
        if os.path.isdir(exp_top_dir+'.bk'):
            shutil.rmtree(exp_top_dir+'.bk')
        shutil.move(exp_top_dir, exp_top_dir+'.bk')        

    checkpoints_path = os.path.join(exp_top_dir, 'ckpt')
    if not os.path.isdir(checkpoints_path):
        os.makedirs(checkpoints_path)
    opt.checkpoints_path = checkpoints_path

    temporal_results_path = os.path.join(exp_top_dir, 'tmp_results')
    if not os.path.isdir(temporal_results_path):
        os.makedirs(temporal_results_path)
    opt.temporal_results_path = temporal_results_path

    desc_log_path = os.path.join(exp_top_dir, 'desc.log')
    with open(desc_log_path, 'w') as f:
        f.write(opt.description)

    train_log_path = os.path.join(exp_top_dir, 'train.log')    
    with open(train_log_path, 'a') as f:
        f.write('Train log for '+opt.train_name+'\n')
    opt.train_log_path = train_log_path

    val_log_path = os.path.join(exp_top_dir, 'val.log')    
    with open(val_log_path, 'a') as f:
        f.write('Val log for '+opt.train_name+'\n')
    opt.val_log_path = val_log_path

    shutil.copytree('../../train', os.path.join(opt.exp_top_dir, 'code'))

    return opt

