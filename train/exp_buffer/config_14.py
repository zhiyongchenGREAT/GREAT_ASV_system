# Generated by run.sh with run_name xvector_cosine_fulltest
import os
print(1)
class Config(object):
    train_name = "small mix 1L AL weight18_1"
    description = 'small mix 1L AL weight step every, no ctrl gamma, gamma 0.2 beta 0.2 fixed, with D class weight 6, \
    long dur, ch adv and rebanlance adv class, just again'
    model = ''
    model_settings = {'in_feat': 30, 'emb_size': 512, 'class_num': 1311, \
    's': 50, 'm': 0.2, 'anneal_steps': 1000, 'weight': 6}
    metric = 'Linear_softmax_ce_head'
    max_step = 1000000

    train_list = '/home/great10/sdsvc_small_100_dataset/mix_1311_smalltrain.csv'
    vox_val_list = '/home/great10/sdsvc_small_100_dataset/source_vox_1211_smallval.csv'
    sdsvc_val_list = '/home/great10/sdsvc_small_100_dataset/target_sdsvc_100_smallval.csv'
    sdsvc_trial_list = '/home/great10/sdsvc_small_100_dataset/sdsvc_trial_list.csv'
    sdsvc_trial_keys = '/home/great10/sdsvc_small_100_dataset/sdsvc_trial_keys.csv'
    vox1test_trial_list = '/home/great10/sdsvc_small_100_dataset/vox1test_trial_list.csv'
    vox1test_trial_keys = '/home/great10/sdsvc_small_100_dataset/vox1test_trial_keys.csv'
    # vox1test_aux_list = '/home/great10/sdsvc_small_100_dataset/vox1test_aux_list.csv'
    # vox1test_aux_keys = '/home/great10/sdsvc_small_100_dataset/vox1test_aux_keys.csv'

    scoring_config = {'p_target': [0.01], 'c_miss': 10, 'c_fa': 1}
    saver_metric = "MINC"
    lr_ctrl = {"metric": "MINC", "Dur": 6, "sig_th": 0.005}
    expect_scheduler_steps = 2

    val_interval_step = 1000

    train_batch_size = 128

    gpu_id = "0, 3"
    num_workers = 32  # how many workers for loading data
    print_freq = 50  # print info every N batch

    lr = 1e-2  # initial learning rate

    weight_decay = 5e-4
    momentum = 0.9

    model_load_path = ''
    cudnn_benchmark = True

    exp_path = "/Lun2/rzz/kaldi-master/egs/zhiyong/sdsvc_exp/train_exp"
    tbx_path = "/Lun2/rzz/kaldi-master/egs/zhiyong/sdsvc_exp/tbx"
