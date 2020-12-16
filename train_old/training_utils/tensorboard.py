import os
import shutil
import tensorboardX

def tensorboard_init(opt):
    if opt.tbx_path == '':
        return None
    writer_path = os.path.join(opt.tbx_path, opt.train_name)
    if os.path.isdir(writer_path):
        # shutil.rmtree(writer_path)
        pass
    writer = tensorboardX.SummaryWriter(log_dir=writer_path)
    return writer
    