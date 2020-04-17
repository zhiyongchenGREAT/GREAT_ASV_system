# import experiment_train
import shutil
import os
import GPU_o
import importlib
import experiment_train

CONFIG_DIR = './config'
EXP_BUFFER = './exp_buffer'

for i in range(0, 1):    
    try:
        shutil.copy(os.path.join(EXP_BUFFER, 'config_'+str(i+1)+'.py'), os.path.join(CONFIG_DIR, 'config.py'))
        shutil.copy(os.path.join(EXP_BUFFER, 'experiment_train_'+str(i+1)+'.py'), os.path.join('./', 'experiment_train.py'))
        importlib.reload(experiment_train)
        experiment_train.main()
    except Exception:
        pass

GPU_o.main("0, 3", 1, 300)