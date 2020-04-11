
def get_epoch_steps(opt, train_data):
    if str(type(train_data)).split('.')[-1][:-2] == 'PickleDataSet_single':
        expected_total_step_epoch = len(train_data) // opt.train_batch_size
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

def stop_ctrl_std(opt, scheduler):
    expect_scheduler_steps = opt.expect_scheduler_steps
    if scheduler.state_dict()['_step_count'] > expect_scheduler_steps:
        return True
    return False
