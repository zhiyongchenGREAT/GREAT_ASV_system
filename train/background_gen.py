import os
import time
import glob
import random
import shutil
import pickle
import traceback
import numpy as np
import pandas as pd
from multiprocessing import Process, Event
from iterable_dataset import VoxIterableDataset

class SpeakerDataGen(object):
    def __init__(self, gen_config = None, data_dir_dict=None, data_len_dict=None, dataset_config=None):
        if gen_config is None:
            self.first_tmp_dir = '/Lun0/zhiyong/dataset/tmp_data/'
            self.second_tmp_dir = '/Lun0/zhiyong/dataset/tmp_data_2/'
            self.log = '/Lun0/zhiyong/dataset/background_gen_log'
            self.workers = 2
            self.tmp_data_csv = '/Lun0/zhiyong/dataset/tmp_data_csv.csv'
        else:
            self.first_tmp_dir = gen_config['first_tmp_dir']
            self.second_tmp_dir = gen_config['second_tmp_dir']
            self.log = gen_config['log']
            self.workers = gen_config['workers']
            self.tmp_data_csv = gen_config['tmp_data_csv']         

        if data_dir_dict is None and data_len_dict is None and dataset_config is None:
            OPT_INDEX = '/Lun0/zhiyong/dataset'
            self.data_dir_dict = {}

            self.data_dir_dict['spk2utt_train_dict'] = os.path.join(OPT_INDEX, 'spk2utt_train_dict')
            self.data_dir_dict['music_dict'] = os.path.join(OPT_INDEX, 'music_dict')
            self.data_dir_dict['noise_dict'] = os.path.join(OPT_INDEX, 'noise_dict')
            self.data_dir_dict['babble_dict'] = os.path.join(OPT_INDEX, 'babble_dict')
            self.data_dir_dict['rir_dict'] = os.path.join(OPT_INDEX, 'rir_dict')

            self.data_len_dict = {}

            self.data_len_dict['spk2utt_train_len'] = os.path.join(OPT_INDEX, 'spk2utt_train_len')
            self.data_len_dict['music_len'] = os.path.join(OPT_INDEX, 'music_len')
            self.data_len_dict['noise_len'] = os.path.join(OPT_INDEX, 'noise_len')
            self.data_len_dict['babble_len'] = os.path.join(OPT_INDEX, 'babble_len')

            self.dataset_config = {}

            self.dataset_config['sr'] = 16000
            self.dataset_config['repeats'] = 150
            self.dataset_config['batch_size'] = 128
            self.dataset_config['extended_prefectch'] = 1.0           
        else:
            self.data_dir_dict = data_dir_dict
            self.data_len_dict = data_len_dict
            self.dataset_config = dataset_config
        
        self.tmp_data_list = pd.read_csv(self.tmp_data_csv, header=None)[0]
        
        self.aux_tmp_data_dict = {}
        for count, i in enumerate(self.tmp_data_list):
            self.aux_tmp_data_dict[count] = i       
        assert len(self.aux_tmp_data_dict) == len(self.tmp_data_list)

    def _processing_code(self, dataset, i, stop_event, log):
        dataset.get_random_list()
        try:
            start_time = time.time()
            for count, (data, label) in enumerate(dataset):
                with open(os.path.join(self.second_tmp_dir, str(i)+str('_')+str(count)), 'wb') as handle:
                    pickle.dump((data.astype(np.float16), label.astype(np.int16)), handle)
                if log is not None:
                    with open(log, 'a') as f:
                        f.write(str(time.time()-start_time)+'\n')               
                start_time = time.time()
                if stop_event.is_set():
                    return
        except Exception:
            with open(log, 'a') as f:
                traceback.print_exc(file=f)
            raise Exception

    def _clear_data(self, clear_dir):
        if os.path.isdir(clear_dir):
            try:
                os.rmdir(clear_dir)
            except OSError:
                print(clear_dir+' Not Empty!')
                shutil.rmtree(clear_dir)
        if not os.path.isdir(clear_dir):
            os.makedirs(clear_dir)

    def setup_and_spawn_process(self):
        self._clear_data(self.second_tmp_dir)
        
        if self.log is not None:
            with open(self.log, 'w') as f:
                pass

        dataset = VoxIterableDataset(self.data_dir_dict, self.data_len_dict, self.dataset_config)
        dataset.noise_data_preload()

        self.processes = []
        self.stop_event = Event()

        for i in range(self.workers):
            p = Process(target = self._processing_code, args = (dataset, i, self.stop_event, self.log))
            self.processes.append(p)

        for p in self.processes:
            p.start() 

    def stop_and_join_process(self):
        print('stoping background gen processes')
        self.stop_event.set()
        joined = [p.join() for p in self.processes]
        del self.processes
        del self.stop_event
    
    def swap_and_clear_data(self):
        new_data_list = glob.glob(os.path.join(self.second_tmp_dir, '*'))

        # reload self.aux_tmp_data_dict if not enough left
        if len(new_data_list) < len(self.aux_tmp_data_dict):
            self.aux_tmp_data_dict = {}
            for count, i in enumerate(self.tmp_data_list):
                self.aux_tmp_data_dict[count] = i       
            assert len(self.aux_tmp_data_dict) == len(self.tmp_data_list)           
        
        available_key_list = list(self.aux_tmp_data_dict.keys())
        random.shuffle(available_key_list)
        swap_key_list = available_key_list[:len(new_data_list)]

        assert len(new_data_list) == len(swap_key_list)

        for (swap_key, new_dir) in zip(swap_key_list, new_data_list):
            tar_dir = self.aux_tmp_data_dict.pop(swap_key)
            assert os.path.isfile(new_dir)
            assert os.path.isfile(tar_dir)
            shutil.move(new_dir, tar_dir)

        self._clear_data(self.second_tmp_dir)
        print('Swap into '+ str(len(new_data_list)) +' data success!')


