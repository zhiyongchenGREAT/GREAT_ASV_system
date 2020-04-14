import experiment_train
import shutil
import os
import GPU_o

CONFIG_DIR = './config'

for i in range(6):
    shutil.move(os.path.join(CONFIG_DIR, 'config_'+str(i+1)+'.py'), os.path.join(CONFIG_DIR, 'config.py'))
    try:
        experiment_train.main()
    except Exception:
        pass

GPU_o.main("0, 3", 1, 400)