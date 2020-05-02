# Generated by run.sh with run_name xvector_cosine_fulltest
import os
print(1)
class Config(object):
    train_name = "small mix 50 1L FOCAL_ALDA_OPT_FAST"
    description = 'small mix 50 1L FOCAL_ALDA_OPT, gamma 0.1 beta 0.1 fixed, class weight 24, \
    long dur, minc lr decay base on vox, f_d 0 f_al 0'
    model = ''
    model_settings = {'in_feat': 30, 'emb_size': 512, 'class_num': 1261, 'source_class_num': 1211, \
    's': 50, 'm': 0.2, 'anneal_steps': 1000, 'weight': 24, 'focal_d_gamma': 0, 'focal_al_gamma': 0}
    metric = ''
    max_step = 1000000

    train_list = '/home/great10/sdsvc_small_50_dataset/mix_1261_smalltrain.csv'
    vox_val_list = '/home/great10/sdsvc_small_50_dataset/source_vox_1211_smallval.csv'
    sdsvc_val_list = '/home/great10/sdsvc_small_50_dataset/target_sdsvc_50_smallval.csv'
    sdsvc_trial_list = '/home/great10/sdsvc_small_50_dataset/sdsvc_trial_list.csv'
    sdsvc_trial_keys = '/home/great10/sdsvc_small_50_dataset/sdsvc_trial_keys.csv'
    vox1test_trial_list = '/home/great10/sdsvc_small_50_dataset/vox1test_trial_list.csv'
    vox1test_trial_keys = '/home/great10/sdsvc_small_50_dataset/vox1test_trial_keys.csv'

    scoring_config = {'p_target': [0.01], 'c_miss': 10, 'c_fa': 1}
    saver_metric = "MINC"
    lr_ctrl = {"metric": "MINC", "Dur": 5, "sig_th": 0.005}
    expect_scheduler_steps = 2

    val_interval_step = 1000

    train_batch_size = 128

    gpu_id = "4, 3"
    num_workers = 32  # how many workers for loading data
    print_freq = 50  # print info every N batch

    lr = 1e-2  # initial learning rate

    weight_decay = 5e-4
    momentum = 0.9

    model_load_path = ''
    cudnn_benchmark = True

    exp_path = "/Lun2/rzz/kaldi-master/egs/zhiyong/sdsvc_exp/train_exp"
    tbx_path = "/Lun2/rzz/kaldi-master/egs/zhiyong/sdsvc_exp/tbx"
