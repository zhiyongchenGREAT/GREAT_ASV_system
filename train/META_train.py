import shutil
import os
import train_x
import importlib
# import experiment_train

CONFIG_DIR = './config'
EXP_BUFFER = '/workspace/LOGS_OUTPUT/std_server5/buffer'
CONFIG_ID = [11, 11]
GPU_O = "4, 5"

for i in range(CONFIG_ID[0], CONFIG_ID[1]+1):    
    try:
        shutil.copy(os.path.join(EXP_BUFFER, 'config_'+str(i)+'.py'), os.path.join(CONFIG_DIR, 'config.py'))
        shutil.copy(os.path.join(EXP_BUFFER, 'experiment_train_'+str(i)+'.py'), os.path.join('./', 'experiment_train.py'))
        import experiment_train
        importlib.reload(experiment_train)
        experiment_train.main()
    except Exception as e:
        print(e)

train_x.main(GPU_O, 1, 200)