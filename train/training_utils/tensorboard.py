import os
import shutil
import tensorboardX

__all__ = ['tensorboard_init']

def tensorboard_init(tbxdir, train_name, **kwargs):
    if tbxdir == '':
        return None
    writer_path = os.path.join(tbxdir, train_name)
    if os.path.isdir(writer_path):
        # shutil.rmtree(writer_path)
        pass
    writer = tensorboardX.SummaryWriter(log_dir=writer_path)
    return writer
    