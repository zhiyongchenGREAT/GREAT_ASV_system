def get_epoch_steps(train_data):
    if str(type(train_data)).split('.')[-1][:-2] == 'PickleDataSet_single':
        expected_total_step_epoch = len(train_data) // opt.train_batch_size
    return expected_total_step_epoch