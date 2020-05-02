# Generated by run.sh with run_name xvector_cosine_fulltest
import os
print(1)
class Config(object):
    train_name = "full mix 1L FOCAL_ALDA_OPT_FAST we6 "
    description = 'full mix 1L FOCAL_ALDA_OPT, gamma 0.1 beta 0.1 fixed, class weight 6, \
    long dur, lr decay base on vox, f_d 0 f_al 0'
    model = ''
    model_settings = {'in_feat': 30, 'emb_size': 512, 'class_num': 7811, 'source_class_num': 7323, \
    's': 50, 'm': 0.2, 'anneal_steps': 10000, 'weight': 6, 'focal_d_gamma': 0, 'focal_al_gamma': 0}
    metric = ''
    max_step = 1000000

    train_list = '/Lun0/zhiyong/sdsvc_full_488_dataset/mix_7811_fulltrain_fixed.csv'
    vox_val_list = '/Lun0/zhiyong/sdsvc_full_488_dataset/source_vox_7323_fullval.csv'
    sdsvc_val_list = '/Lun0/zhiyong/sdsvc_full_488_dataset/target_sdsvc_488_fullval.csv'
    sdsvc_trial_list = '/Lun0/zhiyong/sdsvc_full_488_dataset/sdsvc_trial_list.csv'
    sdsvc_trial_keys = '/Lun0/zhiyong/sdsvc_full_488_dataset/sdsvc_trial_keys.csv'
    vox1test_trial_list = '/Lun0/zhiyong/sdsvc_full_488_dataset/vox1test_trial_list.csv'
    vox1test_trial_keys = '/Lun0/zhiyong/sdsvc_full_488_dataset/vox1test_trial_keys.csv'
    # vox1test_aux_list = '/Lun0/zhiyong/sdsvc_full_488_dataset/vox1test_aux_list.csv'
    # vox1test_aux_keys = '/Lun0/zhiyong/sdsvc_full_488_dataset/vox1test_aux_keys.csv'

    scoring_config = {'p_target': [0.01], 'c_miss': 10, 'c_fa': 1}
    saver_metric = "MINC"
    lr_ctrl = {"metric": "MINC", "Dur": 4, "sig_th": 0.005}
    expect_scheduler_steps = 7

    val_interval_step = 10000

    train_batch_size = 128

    gpu_id = "0, 1"
    num_workers = 128  # how many workers for loading data
    print_freq = 50  # print info every N batch

    lr = 1e-2  # initial learning rate

    weight_decay = 5e-4
    momentum = 0.9

    model_load_path = ''
    cudnn_benchmark = True

    exp_path = "/Lun2/rzz/kaldi-master/egs/zhiyong/sdsvc_exp/train_exp"
    tbx_path = "/Lun2/rzz/kaldi-master/egs/zhiyong/sdsvc_exp/tbx"
