import torch
# import os, kaldi_io
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Sampler, Subset
from torchvision import transforms
import time
import h5py
import random
import pickle

class SRE_CSVDataSet_Tmp(Dataset):
    def __init__(self, spkid2utt_csv):
        self.spkid2utt = pd.read_csv(spkid2utt_csv, header=None)
        
    def __getitem__(self, index):
        spkid = self.spkid2utt[0][index]
        feat_path = self.spkid2utt[1][index]
        feat_path = '/Lun4/SRE/' + feat_path[36:]
        feat = np.load(feat_path)
        feat = feat.astype(np.float32)
        feat = torch.from_numpy(feat)
       
        return feat.squeeze(), spkid

    def __len__(self):
        return len(self.spkid2utt)

class CSVDataSet(Dataset):
    def __init__(self, spkid2utt_csv):
        self.spkid2utt = pd.read_csv(spkid2utt_csv, header=None)
        
    def __getitem__(self, index):
        spkid = self.spkid2utt[0][index]
        feat_path = self.spkid2utt[1][index]
        feat = np.load(feat_path)
        feat = feat.astype(np.float32)
        feat = torch.from_numpy(feat)
       
        return feat.squeeze(), spkid

    def __len__(self):
        return len(self.spkid2utt)

class PickleDataSet(Dataset):
    def __init__(self, spkid2utt_csv):
        self.spkid2utt = pd.read_csv(spkid2utt_csv, header=None)
        
    def __getitem__(self, index):
        path = self.spkid2utt[0][index]
        with open(path, 'rb') as handle:
            batch_data = pickle.load(handle)
        batch_feats = batch_data[0].astype(np.float32)
        batch_labels = batch_data[1]
        if type(batch_labels) is list:
            pass
        elif batch_labels.dtype == np.int16:
            batch_labels = batch_labels.astype(np.int64)
        else:
            raise NotImplementedError
       
        return batch_feats, batch_labels

    def __len__(self):
        return len(self.spkid2utt)

class PickleDataSet_single(Dataset):
    def __init__(self, spkid2utt_csv):
        self.spkid2utt = pd.read_csv(spkid2utt_csv, header=None)
        
    def __getitem__(self, index):
        path = self.spkid2utt[0][index]
        with open(path, 'rb') as handle:
            batch_data = pickle.load(handle)
        batch_feats = batch_data[0].astype(np.float32)
        batch_labels = batch_data[1]
        if type(batch_labels) is list:
            pass
        elif batch_labels.dtype == np.int16 or batch_labels.dtype == np.float16:
            batch_labels = batch_labels.astype(np.int64)
        else:
            raise NotImplementedError
       
        return batch_feats.squeeze(0), batch_labels.squeeze(0)

    def __len__(self):
        return len(self.spkid2utt)

class VoxTrialDataSet(Dataset):
    def __init__(self, trial):
        with open(trial, 'r') as f:
            self.trials = f.readlines()
        
    def __getitem__(self, index):
        trial_label = 1 if self.trials[index].split(' ')[2][:-1] == 'target' else 0
        feat_a_path = self.trials[index].split(' ')[0]
        feat_a = np.load(feat_a_path).astype(np.float32)
        feat_a = torch.tensor(feat_a)
        feat_b_path = self.trials[index].split(' ')[1]
        feat_b = np.load(feat_b_path).astype(np.float32)
        feat_b = torch.tensor(feat_b)
       
        return feat_a, feat_b, trial_label

    def __len__(self):
        return len(self.trials)


class WithReplacementRandomSampler(Sampler):
    """Samples elements randomly, with replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source
        print("Sampler dataset length: ", len(self.data_source))

    def __iter__(self):
        # generate samples of `len(data_source)` that are of value from `0` to `len(data_source)-1`
        samples = np.random.randint(len(self.data_source), size=len(self.data_source))        
        return iter(samples)

    def __len__(self):
        return len(self.data_source)

