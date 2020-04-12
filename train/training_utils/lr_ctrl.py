import os

def vox1test_lr_decay_ctrl(opt, total_step, optimizer, scheduler, train_log):   
    current_lr = optimizer.param_groups[0]['lr']
    scheduler_step = scheduler.state_dict()['_step_count']
    cur_anchor_log = os.path.join(opt.lr_ctrl_path, str(scheduler_step)+".log")
    metric = opt.lr_ctrl["metric"]
    Dur = opt.lr_ctrl["Dur"]
    sig_th = opt.lr_ctrl["sig_th"]

    metric_value = None
    Anchor_Step = None
    Anchor = None
    Count = None

    with open(opt.val_log_path, 'r') as f:
        val_list = f.readlines()

    for i in range(-1, -len(val_list), -1):
        if val_list[i][:-1].split(' ')[0] == 'vox1test_ASV_eval':
            break
        
    line_split = val_list[i][:-1].split(' ')
    for j in range(len(line_split)):
        if line_split[j] == metric+':':
            metric_value = float(line_split[j+1])
        if line_split[j] == 'Lr:':
            if line_split[j+1] != "{:.5f}".format(current_lr):
                metric_value = None
        
    if metric_value is None:
        raise NotImplementedError 

    if os.path.isfile(cur_anchor_log):
        with open(cur_anchor_log, 'r') as f:
            line = f.readlines()[0][:-1]
            line_split = line.split(' ')
            for i in range(len(line_split)):
                if line_split[i] == "Anchor_Step:":
                    Anchor_Step = int(line_split[i+1])
                if line_split[i] == "Anchor:":
                    Anchor = float(line_split[i+1])
                if line_split[i] == "Count:":
                    Count = int(line_split[i+1])
        if Anchor is None or Count is None or Anchor_Step is None:
            raise NotImplementedError

        if metric_value+sig_th >= Anchor:
            msg = "Step: {:} S_Step: {:} Lr: {:.5f} Anchor_Step: {:} Anchor: {:.4f} Count: {:}"\
            .format(total_step, scheduler_step, current_lr, Anchor_Step, Anchor, Count+1)
            with open(cur_anchor_log, 'w') as f:
                f.writelines([msg+'\n'])
            if Count+1 == Dur:
                scheduler.step()
                new_lr = optimizer.param_groups[0]['lr']
                new_scheduler_step = scheduler.state_dict()['_step_count']
                msg = "Step: {:} S_Step: {:} Lr: {:.5f} -> NewLr: {:.5f} New_S_Step: {:}".format(total_step, scheduler_step, \
                current_lr, new_lr, new_scheduler_step)
                with open(cur_anchor_log, 'a') as f:
                    f.writelines([msg+'\n'])
                print(msg)
                train_log.writelines([msg+'\n'])
            return

    msg = "Step: {:} S_Step: {:} Lr: {:.5f} Anchor_Step: {:} Anchor: {:.4f} Count: {:}"\
    .format(total_step, scheduler_step, current_lr, total_step, metric_value, 0)
    with open(cur_anchor_log, 'w') as f:
        f.writelines([msg+'\n']) 
    
    
