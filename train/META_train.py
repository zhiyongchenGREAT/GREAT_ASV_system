import shutil
import os
import GPU_o
import importlib
# import experiment_train

CONFIG_DIR = './config'
EXP_BUFFER = '/Lun2/rzz/kaldi-master/egs/zhiyong/sdsvc_exp/exp_buffer'

for i in range(143, 144):    
    try:
        shutil.copy(os.path.join(EXP_BUFFER, 'config_'+str(i+1)+'.py'), os.path.join(CONFIG_DIR, 'config.py'))
        shutil.copy(os.path.join(EXP_BUFFER, 'experiment_train_'+str(i+1)+'.py'), os.path.join('./', 'experiment_train.py'))
        import experiment_train
        importlib.reload(experiment_train)
        experiment_train.main()
    except Exception as e:
        print(e)

# GPU_o.main("0, 1", 1, 100)