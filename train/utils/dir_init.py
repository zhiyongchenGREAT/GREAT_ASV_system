
def dir_init(opt):

    if opt.model_load_path == '':
        exp_top_dir = os.path.join(opt.exp_path, opt.train_name)
        checkpoints_path = os.path.join(exp_top_dir, 'ckpt')

        if os.path.isdir(opt.exp_top_dir):
            print('Backup the existing train exp dir to ', opt.exp_top_dir+'.bk')
            if os.path.isdir(opt.exp_top_dir+'.bk'):
                shutil.rmtree(opt.exp_top_dir+'.bk')
            shutil.move(opt.exp_top_dir, opt.exp_top_dir+'.bk')
        

        if not os.path.isdir(opt.checkpoints_path):
            os.makedirs(opt.checkpoints_path)
        if not os.path.isdir(opt.train_emb_plot_path):
            os.makedirs(opt.train_emb_plot_path)
        if not os.path.isdir(opt.test_emb_plot_path):
            os.makedirs(opt.test_emb_plot_path)
        if not os.path.isdir(opt.trial_plot_path):
            os.makedirs(opt.trial_plot_path)

        shutil.copy2(opt.config_path, os.path.join(opt.exp_top_dir, 'config.log'))    

        desc_log_path = os.path.join(exp_top_dir, 'desc.log')
        with open(desc_log_path, 'w') as f:
            f.write(opt.description)

        val_log_path = os.path.join(exp_top_dir, 'val.log')    
        with open(val_log_path, 'a') as f:
            f.write('Val log for '+opt.train_name+'\n')
    else:
        shutil.copy2(opt.config_path, os.path.join(opt.exp_top_dir, 'config_'+opt.continue_indicator+'.log'))