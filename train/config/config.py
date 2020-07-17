# Generated by run.sh with run_name xvector_cosine_fulltest
import os

class Config(object):
    train_name = "resnet34_SAP_TAUG_vox"
    description = 'resnet34_SAP_TAUG_vox'
    model = 'Resnet34_SAP'
    model_settings = {'in_feat': 30, 'emb_size': 256, 'class_num': 7323, \
    's': 50, 'm': 0.2, 'anneal_steps': 10000}
    metric = 'AM_normfree_softmax_anneal_ce_head'
    max_step = 1000000

    train_list = '/workspace/DATASET/std/sdsvc_libri_full_dataset/source_vox_7323_fulltrain_fixed.csv'
    vox_val_list = '/workspace/DATASET/std/sdsvc_libri_full_dataset/source_vox_7323_fullval.csv'
    libri_val_list = '/workspace/DATASET/std/sdsvc_libri_full_dataset/source_libri_1172_fullval.csv'
    sdsvc_val_list = '/workspace/DATASET/std/sdsvc_libri_full_dataset/target_sdsvc_488_fullval.csv'
    sdsvc_trial_list = '/workspace/DATASET/std/sdsvc_libri_full_dataset/sdsvc_trial_list.csv'
    sdsvc_trial_keys = '/workspace/DATASET/std/sdsvc_libri_full_dataset/sdsvc_trial_keys.csv'
    vox1test_trial_list = '/workspace/DATASET/std/sdsvc_libri_full_dataset/vox1test_trial_list.csv'
    vox1test_trial_keys = '/workspace/DATASET/std/sdsvc_libri_full_dataset/vox1test_trial_keys.csv'
    libri_trial_list = '/workspace/DATASET/std/sdsvc_libri_full_dataset/libri_trial_list.csv'
    libri_trial_keys = '/workspace/DATASET/std/sdsvc_libri_full_dataset/libri_trial_keys.csv'
    # vox1test_aux_list = '/workspace/DATASET/std/sdsvc_libri_full_dataset/vox1test_aux_list.csv'
    # vox1test_aux_keys = '/workspace/DATASET/std/sdsvc_libri_full_dataset/vox1test_aux_keys.csv'

    scoring_config = {'p_target': [0.01, 0.005], 'c_miss': 1, 'c_fa': 1}
    saver_metric = "MINC"
    lr_ctrl = {"metric": "MINC", "Dur": 4, "sig_th": 0.005}
    expect_scheduler_steps = 7

    val_interval_step = 10000

    train_batch_size = 128

    gpu_id = "0, 1"
    num_workers = 32  # how many workers for loading data
    print_freq = 50  # print info every N batch

    lr = 1e-2  # initial learning rate

    weight_decay = 5e-4
    momentum = 0.9

    model_load_path = ''
    cudnn_benchmark = True

    exp_path = "/workspace/LOGS_OUTPUT/std_server5/train"
    tbx_path = "/workspace/LOGS_OUTPUT/std_server5/tbx"
