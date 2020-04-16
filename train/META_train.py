# import experiment_train
import shutil
import os
import GPU_o
import importlib
import experiment_train

CONFIG_DIR = './config'

for i in range(3):    
    try:
        shutil.copy(os.path.join(CONFIG_DIR, 'config_'+str(i+1)+'.py'), os.path.join(CONFIG_DIR, 'config.py'))
        importlib.reload(experiment_train)
        experiment_train.main()
    except Exception:
        pass

GPU_o.main("0, 3", 1, 400)