
def get_epoch_steps(opt, train_data):
    if str(type(train_data)).split('.')[-1][:-2] == 'PickleDataSet_single':
        expected_total_step_epoch = len(train_data) // opt.train_batch_size
    elif str(type(train_data)).split('.')[-1][:-2] == 'PickleDataSet':
        expected_total_step_epoch = len(train_data)
    else:
        raise NotImplementedError
    
    return expected_total_step_epoch

def standard_freq_logging(delta_time, total_step, train_loss, train_acc, optimizer, train_log, tbx_writer):
    current_lr = optimizer.param_groups[0]['lr'] # assuming normal optimizer all params use same lr

    msg = "Dur: {:.3f} Step: {:} TrLoss: {:.4f} TrAcc: {:.4f} Lr :{:.6f}"\
    .format(delta_time, total_step, train_loss, train_acc, current_lr)
    print(msg)
    train_log.writelines([msg+'\n'])

    tbx_writer.add_scalar('Trainloss', train_loss, total_step)
    tbx_writer.add_scalar('TrainAcc', train_acc, total_step)
    tbx_writer.add_scalar('Lr', current_lr, total_step)

def da_freq_logging(delta_time, total_step, train_loss, train_acc, optimizer, train_log, tbx_writer):
    current_lr = optimizer.param_groups[0]['lr'] # assuming normal optimizer all params use same lr

    train_loss_c = train_loss[0]
    train_loss_d = train_loss[1]
    train_loss_al = train_loss[2]
    
    train_acc_c = train_acc[0]
    train_acc_d = train_acc[1]

    msg = "Dur: {:.3f} Step: {:} TrLoss: {:.4f} TrAcc: {:.4f} Lr :{:.6f}"\
    .format(delta_time, total_step, train_loss_c, train_acc_c, current_lr)
    print(msg)
    train_log.writelines([msg+'\n'])

    tbx_writer.add_scalar('Trainloss', train_loss_c, total_step)
    tbx_writer.add_scalar('Trainloss_d', train_loss_d, total_step)
    tbx_writer.add_scalar('Trainloss_al', train_loss_al, total_step)

    tbx_writer.add_scalar('TrainAcc', train_acc_c, total_step)
    tbx_writer.add_scalar('TrainAcc_d', train_acc_d, total_step)

    tbx_writer.add_scalar('Lr', current_lr, total_step)    

def stop_ctrl_std(opt, scheduler):
    expect_scheduler_steps = opt.expect_scheduler_steps
    if scheduler.state_dict()['_step_count'] > expect_scheduler_steps:
        return True
    return False
