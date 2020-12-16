#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys, time, os, argparse, socket, importlib
import yaml
import numpy
import pdb
import torch
import glob
from tuneThreshold import tuneThresholdfromScore_std
# from SpeakerNet import SpeakerNet
from DatasetLoader import get_data_loader
import os
import shutil
import training_utils
import fitlog


parser = argparse.ArgumentParser(description = "SpeakerNet");

parser.add_argument('--config',         type=str,   default=None,   help='Config YAML file');

## Data loader
parser.add_argument('--max_frames',     type=int,   default=300,    help='Input length to the network for training');
parser.add_argument('--eval_frames',    type=int,   default=0,    help='Input length to the network for testing; 0 uses the whole files');
parser.add_argument('--batch_size',     type=int,   default=128,    help='Batch size, number of speakers per batch');
parser.add_argument('--max_seg_per_spk', type=int,  default=100,    help='Maximum number of utterances per speaker per epoch');
parser.add_argument('--nDataLoaderThread', type=int, default=5,     help='Number of loader threads');
parser.add_argument('--augment',        type=bool,  default=True,  help='Augment input')

## Training details
parser.add_argument('--test_interval',  type=int,   default=20,     help='Test and save every [test_interval] epochs');
parser.add_argument('--max_epoch',      type=int,   default=200,    help='Maximum number of epochs');
parser.add_argument('--trainfunc',      type=str,   default="",     help='Loss function');

## Optimizer
parser.add_argument('--optimizer',      type=str,   default="sgd", help='sgd or adam');
parser.add_argument('--scheduler',      type=str,   default="cosine", help='Learning rate scheduler');
parser.add_argument('--lr_step',        type=str,   default="iteration", help='Learning rate scheduler');
parser.add_argument('--lr',             type=float, default=0.01,  help='Learning rate');
parser.add_argument('--base_lr',        type=float, default=1e-5,  help='Learning rate min');
parser.add_argument('--cycle_step',     type=int, default=130000,  help='Learning rate cycle');
parser.add_argument('--expected_step',  type=int, default=520000,  help='Total steps');
parser.add_argument("--lr_decay",       type=float, default=0.25,   help='Learning rate decay every [test_interval] epochs');
parser.add_argument('--weight_decay',   type=float, default=5e-4,      help='Weight decay in the optimizer');

## Loss functions
parser.add_argument("--hard_prob",      type=float, default=None,    help='Hard negative mining probability, otherwise random, only for some loss functions');
parser.add_argument("--hard_rank",      type=int,   default=None,     help='Hard negative mining rank in the batch, only for some loss functions');
parser.add_argument('--margin',         type=float, default=0.2,      help='Loss margin, only for some loss functions');
parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions');
parser.add_argument('--nPerSpeaker',    type=int,   default=1,      help='Number of utterances per speaker per batch, only for metric learning based losses');
parser.add_argument('--nClasses',       type=int,   default=5994,   help='Number of speakers in the softmax layer, only for softmax-based losses');

## Load and save
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights');
# parser.add_argument('--save_path',      type=str,   default="", help='Path for model and logs');

## Training and test data
parser.add_argument('--train_list',     type=str,   default="/workspace/DATASET/server9_ssd/voxceleb/vox2_trainlist.txt",     help='Train list');
parser.add_argument('--test_list',      type=str,   default="/workspace/DATASET/server9_ssd/voxceleb/vox_o_triallist.txt",     help='Evaluation list');
parser.add_argument('--enroll_list',    type=str,   default="",     help='Enroll list');
parser.add_argument('--train_path',     type=str,   default="/workspace/DATASET/server9_ssd/voxceleb", help='Absolute path to the train set');
parser.add_argument('--test_path',      type=str,   default="/workspace/DATASET/server9_ssd/voxceleb", help='Absolute path to the test set');
parser.add_argument('--musan_path',     type=str,   default="/workspace/DATASET/server9_ssd/musan_split", help='Absolute path to the test set');
parser.add_argument('--rir_path',       type=str,   default="/workspace/DATASET/server9_ssd/RIRS_NOISES/simulated_rirs", help='Absolute path to the test set');

# ## Training and test data
# parser.add_argument('--train_list',     type=str,   default="/workspace/DATASET/server9_ssd/train_list.txt",     help='Train list');
# parser.add_argument('--test_list',      type=str,   default="/workspace/DATASET/server9_ssd/voxceleb/voxsrc2020_final_pairs.txt",     help='Evaluation list');
# parser.add_argument('--enroll_list',    type=str,   default="",     help='Enroll list');
# parser.add_argument('--train_path',     type=str,   default="/workspace/DATASET/server9_ssd/voxceleb", help='Absolute path to the train set');
# parser.add_argument('--test_path',      type=str,   default="/workspace/DATASET/server9_ssd/voxceleb/voxsrc2020", help='Absolute path to the test set');
# parser.add_argument('--musan_path',     type=str,   default="/workspace/DATASET/server9_ssd/musan_split", help='Absolute path to the test set');
# parser.add_argument('--rir_path',       type=str,   default="/workspace/DATASET/server9_ssd/RIRS_NOISES/simulated_rirs", help='Absolute path to the test set');

## Model definition
parser.add_argument('--n_mels',         type=int,   default=40,     help='Number of mel filterbanks');
parser.add_argument('--log_input',      type=bool,  default=True,  help='Log input features')
parser.add_argument('--model',          type=str,   default="",     help='Name of model definition');
parser.add_argument('--encoder_type',   type=str,   default="",  help='Type of encoder');
parser.add_argument('--nOut',           type=int,   default=192,    help='Embedding size in the last FC layer');

## Training Control
parser.add_argument('--trainlogs',      type=str,   default="/workspace/LOGS_OUTPUT/tmp_logs/train_logs_201120");
parser.add_argument('--fitlogdir',      type=str,   default="/workspace/LOGS_OUTPUT/tmp_logs/ASV_LOGS_201120");
parser.add_argument('--tbxdir',         type=str,   default="/workspace/LOGS_OUTPUT/tmp_logs/tbx")
parser.add_argument('--fitlog_DATASET', type=str,   default="otf_vox2_aug");
parser.add_argument('--fitlog_Desc',    type=str,   default="vox2_newsystem_base_epacatdnn");
parser.add_argument('--train_name',     type=str,   default="vox2_newsystem_base_epacatdnn");
parser.add_argument('--amp',            type=bool,  default=True);
parser.add_argument('--GPU',            type=str,   default="4");


## For test only
parser.add_argument('--distance_m',     type=str, default="cosine", help='Eval distance metric')
parser.add_argument('--eval', dest='eval', action='store_true', help='Eval only')

args = parser.parse_args();

## Parse YAML
def find_option_type(key, parser):
    for opt in parser._get_optional_actions():
        if ('--' + key) in opt.option_strings:
           return opt.type
    raise ValueError

if args.config is not None:
    with open(args.config, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yml_config.items():
        if k in args.__dict__:
            typ = find_option_type(k, parser)
            args.__dict__[k] = typ(v)
        else:
            sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))

## Set visible CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

## Initialise directories
model_save_path     = os.path.join(args.trainlogs, args.train_name, "model")
result_save_path    = os.path.join(args.trainlogs, args.train_name, "result")


if os.path.isdir(os.path.join(args.trainlogs, args.train_name)):
    shutil.rmtree(os.path.join(args.trainlogs, args.train_name))

if not(os.path.exists(model_save_path)):
    os.makedirs(model_save_path)
        
if not(os.path.exists(result_save_path)):
    os.makedirs(result_save_path)

## backup train dir
train_file_dir = os.path.dirname(os.path.realpath(__file__))
shutil.copytree(train_file_dir, os.path.join(args.trainlogs, args.train_name, 'code'))

## fitlog
training_utils.standard_fitlog_init(**vars(args))

## tb
tbxwriter = training_utils.tensorboard_init(**vars(args))

## Load models
if not args.amp:
    SpeakerNet = importlib.import_module('SpeakerNet').__getattribute__('SpeakerNet')
else:
    SpeakerNet = importlib.import_module('SpeakerNet_amp').__getattribute__('SpeakerNet')

s = SpeakerNet(tbxwriter=tbxwriter, **vars(args))

it          = 1;
prevloss    = float("inf");
sumloss     = 0;
min_eer     = [100];

## Load model weights
modelfiles = glob.glob('%s/model0*.model'%model_save_path)
modelfiles.sort()

if len(modelfiles) >= 1:
    s.loadParameters(modelfiles[-1], only_para=False);
    print("Model %s loaded from previous state!"%modelfiles[-1]);
    # Note iteration is renewed to next, but total_step inherits from previous
    it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1
elif(args.initial_model != ""):
    ## use new load models
    # s.loadParameters(args.initial_model, only_para=True);
    ## use old load models
    s.loadParameters_old(args.initial_model, only_para=True);
    print("Model %s loaded!"%args.initial_model);        


## Write args to scorefile
scorefile = open(result_save_path+"/scores.txt", "a+");

for items in vars(args):
    print(items, vars(args)[items])
    scorefile.write('%s %s\n'%(items, vars(args)[items]))
scorefile.flush()

print('\n')

## Evaluation code
if args.eval == True:
    if args.enroll_list == '':   
        sc, lab, trials = s.evaluateFromList(args.test_list, distance_m=args.distance_m, print_interval=100, \
        test_path=args.test_path, eval_frames=args.eval_frames)
    else:
        sc, lab, trials = s.evaluateFromListAndDict(listfilename=args.test_list, enrollfilename=args.enroll_list, \
        distance_m=args.distance_m, print_interval=100, \
        test_path=args.test_path, eval_frames=args.eval_frames)

    result = tuneThresholdfromScore_std(sc, lab);
    print('EER %2.4f MINC@0.01 %.5f MINC@0.001 %.5f'%(result[1], result[-2], result[-1]))
    scorefile.write('\nEER %2.4f MINC@0.01 %.5f MINC@0.001 %.5f\n'%(result[1], result[-2], result[-1]))
    scorefile.flush()

    fitlog.add_best_metric({"Voxceleb_O":{"EER":result[1]}})
    fitlog.add_best_metric({"Voxceleb_O":{"MINC_0.01":result[-2]}})
    fitlog.add_best_metric({"Voxceleb_O":{"MINC_0.001":result[-1]}})

    ## Save scores
    print('Auto save scores to results dir.')
    with open(result_save_path+"/eval_scores.txt",'w') as outfile:
        for vi, val in enumerate(sc):
            outfile.write('%.4f %s\n'%(val,trials[vi]))    
    quit()

## Initialise data loader
trainLoader = get_data_loader(args.train_list, **vars(args))

while(1):   

    clr = [x['lr'] for x in s.__optimizer__.param_groups]

    print(time.strftime("%Y-%m-%d %H:%M:%S"), it, "Training %s with LR %f..."%(args.model,max(clr)))

    ## Train network
    loss, traineer, stop = s.train_network(loader=trainLoader)

    ## Validate and save
    if it % args.test_interval == 0 or stop == True:

        print(time.strftime("%Y-%m-%d %H:%M:%S"), it, "Evaluating...")

        sc, lab, trials = s.evaluateFromList(args.test_list, distance_m=args.distance_m, print_interval=100, \
        test_path=args.test_path, eval_frames=args.eval_frames)

        # sc, lab, trials = s.evaluateFromListAndDict(listfilename=args.test_list, enrollfilename=args.enroll_list, \
        # distance_m=arg.distance_m, print_interval=100, \
        # test_path=args.test_path, eval_frames=args.eval_frames)

        result = tuneThresholdfromScore_std(sc, lab)

        min_eer.append(result[1])

        print(time.strftime("%Y-%m-%d %H:%M:%S"), "LR %f, TEER/TAcc %2.2f, TLOSS %f, VEER %2.4f, MINEER %2.4f"%( max(clr), traineer, loss, result[1], min(min_eer)))
        scorefile.write("IT %d, LR %f, TEER/TAcc %2.2f, TLOSS %f, VEER %2.4f, MINEER %2.4f\n"%(it, max(clr), traineer, loss, result[1], min(min_eer)))
        scorefile.flush()

        ## Add fitlog
        training_utils.vox1_o_ASV_step_fitlog(result[1], result[-2], result[-1], it)

        ## If best add best fitlog and log scores
        if result[1] == min(min_eer):
            training_utils.vox1_o_ASV_best_fitlog(result[1], result[-2], result[-1])
            
            with open(result_save_path+"/model%09d.vox1osc"%it,'w') as outfile:
                for vi, val in enumerate(sc):
                    outfile.write('%.4f %s\n'%(val,trials[vi]))

        s.saveParameters(model_save_path+"/model%09d.model"%it)
        
        with open(model_save_path+"/model%09d.eer"%it, 'w') as eerfile:
            eerfile.write('%.4f'%result[1])

        if stop == True:
            quit()

    else:
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "LR %f, TEER/TAcc %2.2f, TLOSS %f"%( max(clr), traineer, loss))
        scorefile.write("IT %d, LR %f, TEER/TAcc %2.2f, TLOSS %f\n"%(it, max(clr), traineer, loss))

        scorefile.flush()

    if it >= args.max_epoch:
        quit()

    it+=1
    print("")

scorefile.close()
