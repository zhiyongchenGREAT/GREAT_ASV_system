#!/usr/bin/python
#-*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy, math, pdb, sys, random
import time, os, itertools, shutil, importlib
from tuneThreshold import tuneThresholdfromScore
from DatasetLoader import loadWAV

from torch.cuda.amp import autocast, GradScaler

class WrappedModel(nn.Module):

    ## The purpose of this wrapper is to make the model structure consistent between single and multi-GPU

    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x, label=None, total_step=None):
        return self.module(x, label, total_step)


class SpeakerNet(nn.Module):

    def __init__(self, model, optimizer, trainfunc, nPerSpeaker, expected_step, Syncbatch=False, **kwargs):
        super(SpeakerNet, self).__init__();

        SpeakerNetModel = importlib.import_module('models.'+model).__getattribute__('MainModel')
        self.__S__ = SpeakerNetModel(**kwargs)

        SpeakerNetModel = importlib.import_module('models.X_vector').__getattribute__('MainModel')
        self.__Ss__ = SpeakerNetModel(**kwargs)

        if Syncbatch:
            raise NotImplementedError

        LossFunction = importlib.import_module('loss.'+trainfunc).__getattribute__('LossFunction')
        
        self.__L__ = LossFunction(**kwargs)
        if Syncbatch:
            raise NotImplementedError

        self.nPerSpeaker = nPerSpeaker

        ## MLP layers
        nOut = kwargs['nOut']
        self.embedding_layer1 = torch.nn.Sequential()
        self.embedding_layer1.add_module('linear', nn.Linear(nOut, nOut))
        self.embedding_layer1.add_module('relu', nn.ReLU(True))
        self.embedding_layer1.add_module('batchnorm',nn.BatchNorm1d(nOut))

        self.embedding_layer2 = torch.nn.Sequential()
        self.embedding_layer2.add_module('linear', nn.Linear(nOut, nOut))
        self.embedding_layer2.add_module('batchnorm',nn.BatchNorm1d(nOut))

        ## sidetune para
        self.expected_step = expected_step
        self.alpha_min = 0.25 # side weight 0.75
        self.k = self.expected_step * (self.alpha_min) / (1.0 - self.alpha_min)

    def forward(self, data, label=None, total_step=None):

        if total_step is None:
            self.alpha = self.alpha_min
        else:
            self.alpha = self.k / (self.k + total_step)

        data    = data.reshape(-1,data.size()[-1]).cuda()
        outp    = self.alpha*self.__S__.forward(data) + (1.0-self.alpha)*self.__Ss__.forward(data)

        outp    = self.embedding_layer1(outp)
        outp    = self.embedding_layer2(outp)

        if label == None:
            return outp

        else:

            outp    = outp.reshape(self.nPerSpeaker,-1,outp.size()[-1]).transpose(1,0).squeeze(1)

            nloss, prec1 = self.__L__.forward(outp,label)

            return nloss, prec1, self.alpha


class ModelTrainer(object):

    def __init__(self, speaker_model, optimizer, scheduler, gpu, mixedprec, tbxwriter=None, **kwargs):

        self.__model__  = speaker_model

        Optimizer = importlib.import_module('optimizer.'+optimizer).__getattribute__('Optimizer')
        self.__optimizer__ = Optimizer(list(self.__model__.module.__Ss__.parameters())+list(self.__model__.module.__L__.parameters()), **kwargs)

        Scheduler = importlib.import_module('scheduler.'+scheduler).__getattribute__('Scheduler')
        self.__scheduler__, self.lr_step, self.expected_step = Scheduler(self.__optimizer__, **kwargs)

        self.scaler = GradScaler() 

        self.gpu = gpu

        self.mixedprec = mixedprec

        assert self.lr_step in ['epoch', 'iteration']
        self.total_step = 0
        self.stop = False
        self.tbxwriter = tbxwriter

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Train network
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, loader, verbose):

        self.__model__.train();

        stepsize = loader.batch_size;

        counter = 0
        index   = 0
        loss    = 0
        top1    = 0     # EER or accuracy

        tstart = time.time()
        
        for data, data_label in loader:

            data    = data.transpose(1,0)

            self.__model__.zero_grad()

            label   = torch.LongTensor(data_label).cuda()

            if self.mixedprec:
                with autocast():
                    nloss, prec1, self.alpha = self.__model__(data, label, self.total_step)
                self.scaler.scale(nloss).backward()
                self.scaler.step(self.__optimizer__)
                self.scaler.update()  
            else:
                raise NotImplementedError

            loss    += nloss.detach().cpu()
            top1    += prec1.detach().cpu()          
            counter += 1
            index   += stepsize

            telapsed = time.time() - tstart
            tstart = time.time()

            if verbose:
                clr = [x['lr'] for x in self.__optimizer__.param_groups]
                sys.stdout.write("\rGPU (%d) Total_step (%d) Processing (%d) "%(self.gpu, self.total_step, index))
                sys.stdout.write("Loss %f Lr %.5f TEER/TAcc %2.3f%% Alph %2.3f - %.2f Hz "%(loss/counter, max(clr), top1/counter, self.alpha, stepsize/telapsed))
                sys.stdout.flush()
                self.tbxwriter.add_scalar('Trainloss', nloss.detach().cpu(), self.total_step)
                self.tbxwriter.add_scalar('TrainAcc', prec1, self.total_step)
                self.tbxwriter.add_scalar('Lr', max(clr), self.total_step)
            if self.lr_step == 'iteration': self.__scheduler__.step()

            self.total_step += 1

            if self.total_step >= self.expected_step:
                self.stop = True
                print('')
                return (loss/counter, top1/counter, self.stop)

        if self.lr_step == 'epoch': self.__scheduler__.step()

        print('')
        
        return (loss/counter, top1/counter, self.stop)


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def evaluateFromList(self, listfilename, distance_m='L2', print_interval=100, test_path='', num_eval=10, eval_frames=None, verbose=True):
        assert distance_m in ['L2', 'cosine']
        if verbose:
            print('Distance metric: %s'%(distance_m))
            print('Evaluating from trial file: %s'%(listfilename))

        self.__model__.eval()
        
        lines       = []
        files       = []
        feats       = {}
        tstart      = time.time()

        ## Read all lines
        with open(listfilename) as listfile:
            while True:
                line = listfile.readline()
                if (not line):
                    break

                data = line.split();

                ## Append random label if missing
                if len(data) == 2: data = [random.randint(0,1)] + data

                files.append(data[1])
                files.append(data[2])
                lines.append(line)

        setfiles = list(set(files))
        setfiles.sort()

        ## Save all features to file
        for idx, file in enumerate(setfiles):

            inp1 = torch.FloatTensor(loadWAV(os.path.join(test_path,file), eval_frames, evalmode=True, num_eval=num_eval)).cuda()

            ref_feat = self.__model__.forward(inp1).detach().cpu()

            filename = '%06d.wav'%idx

            feats[file]     = ref_feat

            telapsed = time.time() - tstart

            if (idx % print_interval == 0) and verbose:
                sys.stdout.write("\rReading %d of %d: %.2f Hz, embedding size %d"%(idx,len(setfiles),idx/telapsed,ref_feat.size()[1]))


        all_scores = []
        all_labels = []
        all_trials = []
        tstart = time.time()

        ## Read files and compute all scores
        for idx, line in enumerate(lines):

            data = line.split()

            ## Append random label if missing
            if len(data) == 2: data = [random.randint(0,1)] + data


            ref_feat = feats[data[1]]
            com_feat = feats[data[2]]
            # ref_feat = (feats[data[1]] - mean_vector).cuda() 
            # com_feat = (feats[data[2]] - mean_vector).cuda()

            # if self.__model__.module.__L__.test_normalize:
            ref_feat = F.normalize(ref_feat, p=2, dim=1)
            com_feat = F.normalize(com_feat, p=2, dim=1)

            if distance_m == 'L2':
                dist = F.pairwise_distance(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).numpy()
                score = -1 * numpy.mean(dist)
            elif distance_m == 'cosine':
                dist = F.cosine_similarity(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).numpy()
                score = numpy.mean(dist)

            all_scores.append(score)
            all_labels.append(int(data[0]))
            all_trials.append(data[1]+" "+data[2])

            if (idx % (print_interval*100) == 0) and verbose:
                telapsed = time.time() - tstart
                sys.stdout.write("\rComputing %d of %d: %.2f Hz"%(idx,len(lines),idx/telapsed))
                sys.stdout.flush()

        print('')

        return (all_scores, all_labels, all_trials);

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list and dict
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def evaluateFromListAndDict(self, listfilename, enrollfilename, distance_m, print_interval=100, \
        test_path='', num_eval=10, eval_frames=None, verbose=True):
        
        assert distance_m in ['L2', 'cosine']
        if verbose:
            print('Distance metric: %s'%(distance_m))
            print('Evaluating from trial file: %s'%(listfilename))
            print('Enroll from file: %s'%(enrollfilename))
        
        self.__model__.eval()
        
        trial_lines        = []
        trial_files        = []
        enroll_files       = {}
        enroll_feats       = {}
        trial_feats         = {}

        ## Read all enroll lines
        with open(enrollfilename) as listfile:
            while True:
                line = listfile.readline()
                if (not line):
                    break

                data = line.split();

                ## Enroll file should have line length >= 2
                assert len(data) >= 2

                enroll_files[data[0]] = data[1:]

        ## Extract all features to enroll_feats
        tstart = time.time()

        for idx, enroll_id in enumerate(enroll_files):
            for file in enroll_files[enroll_id]:

                inp1 = torch.FloatTensor(loadWAV(os.path.join(test_path,file), eval_frames, evalmode=True, num_eval=num_eval)).cuda()

                ref_feat = self.__model__.forward(inp1).detach().cpu()
                
                if enroll_id not in enroll_feats.keys():
                    enroll_feats[enroll_id] = ref_feat
                else:
                    enroll_feats[enroll_id] = torch.cat([enroll_feats[enroll_id], ref_feat], axis=0)

            telapsed = time.time() - tstart

            if (idx % print_interval == 0) and verbose:
                sys.stdout.write("\rEnroll Reading %d of %d: %.2f Hz, embedding size %d"%(idx,len(enroll_files),idx/telapsed,ref_feat.size()[1]))
                sys.stdout.flush()

        ## Read all trial lines
        with open(listfilename) as listfile:
            while True:
                line = listfile.readline()
                if (not line):
                    break

                data = line.split();

                ## Append random label if missing
                if len(data) == 2: data = [random.randint(0,1)] + data

                trial_files.append(data[2])
                trial_lines.append(line)

        setfiles = list(set(trial_files))
        setfiles.sort()

        ## Extract all features to trial_feats
        tstart = time.time()

        for idx, file in enumerate(setfiles):

            inp1 = torch.FloatTensor(loadWAV(os.path.join(test_path,file), eval_frames, evalmode=True, num_eval=num_eval)).cuda()

            ref_feat = self.__model__.forward(inp1).detach().cpu()

            trial_feats[file]     = ref_feat

            telapsed = time.time() - tstart

            if (idx % print_interval == 0) and verbose:
                sys.stdout.write("\rReading %d of %d: %.2f Hz, embedding size %d"%(idx,len(setfiles),idx/telapsed,ref_feat.size()[1]))
                sys.stdout.flush()

        all_scores = []
        all_labels = []
        all_trials = []
        tstart = time.time()


        ## Read files and compute all scores
        for idx, line in enumerate(trial_lines):

            data = line.split()

            ## Append random label if missing
            if len(data) == 2: data = [random.randint(0,1)] + data

            # ref_feat = enroll_feats[data[1]].cuda()

            ref_feat = torch.mean(enroll_feats[data[1]], axis=0, keepdim=True)
            com_feat = trial_feats[data[2]]

            # if self.__model__.module.__L__.test_normalize:
            ref_feat = F.normalize(ref_feat, p=2, dim=1)
            com_feat = F.normalize(com_feat, p=2, dim=1)

            if distance_m == 'L2':
                dist = F.pairwise_distance(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).numpy()
                score = -1 * numpy.mean(dist)
            elif distance_m == 'cosine':
                dist = F.cosine_similarity(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).numpy()
                score = numpy.mean(dist)

            all_scores.append(score)
            all_labels.append(int(data[0]))
            all_trials.append(data[1]+" "+data[2])

            if (idx % (print_interval*100) == 0) and verbose:
                telapsed = time.time() - tstart
                sys.stdout.write("\rComputing %d of %d: %.2f Hz"%(idx, len(trial_lines), idx/telapsed))
                sys.stdout.flush()

        print('')

        return (all_scores, all_labels, all_trials)


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveParameters(self, path):
        state = {
            'model': self.__model__.module.state_dict(),
            'optimizer': self.__optimizer__.state_dict(),
            'scheduler': self.__scheduler__.state_dict(),
            'scaler': self.scaler.state_dict(),
            'total_step': self.total_step
            }       
        torch.save(state, path)


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====
    def loadParameters_sidetune(self, path_S, path_Ss, only_para=False):

        self_state = self.__model__.module.state_dict()
        loaded_state = torch.load(path_S, map_location="cuda:%d"%self.gpu)
        for name, param in loaded_state['model'].items():
            origname = name

            ## pass spk clf weight
            if '__L__' in name:
                print('pass __L__ classerfier W')
                continue

            ## pass DA weight
            if 'DA_module' in name:
                print('pass DA_module params:'+name)
                continue

            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("#%s is not in the model."%origname)
                    continue

            if self_state[name].size() != loaded_state['model'][origname].size():
                print("#Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state['model'][origname].size()))
                continue

            self_state[name].copy_(param)

        print('#######################Side params###################')
        ## load sidenet params
        self_state = self.__model__.module.state_dict()
        loaded_state = torch.load(path_Ss, map_location="cuda:%d"%self.gpu)
        for name, param in loaded_state['model'].items():
            origname = name         
            if '__S__' in origname:
                name = name.replace("__S__", "__Ss__")

            ## pass spk clf weight
            if '__L__' in name:
                print('pass __L__ classerfier W')
                continue

            ## pass DA weight
            if 'DA_module' in name:
                print('pass DA_module params:'+name)
                continue
            
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("#%s is not in the model."%origname)
                    continue

            if self_state[name].size() != loaded_state['model'][origname].size():
                print("#Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state['model'][origname].size()))
                continue

            self_state[name].copy_(param)

        if not only_para:    
            # loaded_state = loaded_state['optimizer']
            self.__optimizer__.load_state_dict(loaded_state['optimizer'])

            loaded_state['scheduler']['last_epoch'] = loaded_state['scheduler']['last_epoch'] - 1
            loaded_state['scheduler']['_step_count'] = loaded_state['scheduler']['_step_count'] - 1

            print('#Scheduler -1 last_e: %d step_count: %d'%\
            (loaded_state['scheduler']['last_epoch'], loaded_state['scheduler']['_step_count']))

            self.__scheduler__.load_state_dict(loaded_state['scheduler'])
            self.__scheduler__.step()

            self.scaler.load_state_dict(loaded_state["scaler"])

            self.total_step = loaded_state['total_step']
            print('#Resume from step: %d'%(self.total_step))
        else:
            print('#Only params are loaded, start from beginning...')


    def loadParameters(self, path, only_para=False):

        self_state = self.__model__.module.state_dict()
        loaded_state = torch.load(path, map_location="cuda:%d"%self.gpu)
        # loaded_state = torch.load(path, map_location="cpu")

        for name, param in loaded_state['model'].items():
            origname = name

            ## pass spk clf weight
            if '__L__' in name:
                print('pass __L__ classerfier W')
                continue

            ## pass DA weight
            if 'DA_module' in name:
                print('pass DA_module params:'+name)
                continue

            if name not in self_state:
                name = name.replace("module.", "")

                if name not in self_state:
                    print("#%s is not in the model."%origname)
                    continue

            if self_state[name].size() != loaded_state['model'][origname].size():
                print("#Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state['model'][origname].size()))
                continue

            self_state[name].copy_(param)

        if not only_para:    
            # loaded_state = loaded_state['optimizer']
            print('#Resume not available')
            raise
            
        else:
            print('#Only params are loaded, start from beginning...')