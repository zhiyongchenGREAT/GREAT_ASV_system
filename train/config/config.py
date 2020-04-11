# Generated by run.sh with run_name xvector_cosine_fulltest
import os

class Config(object):
    train_name = "new_pipe_test"
    description = 'new_pipe_test'
    model = 'Xvector_SAP'
    model_settings = {'in_feat': 30, 'emb_size': 512, 'class_num': 1311, \
    's': 50, 'm': 0.2, 'anneal_steps': 1000}
    metric = 'AM_normfree_softmax_anneal_ce_head'
    max_step = 100000


    train_list = '/Lun0/zhiyong/sdsvc_small_100_dataset/mix_1311_smalltrain.csv'
    # train_list = '/Lun0/zhiyong/vox_small_evad_sparse/vox_small_evad_sparse.csv'
    # train_list = '/Lun0/zhiyong/sdsvc_t2_small_sparse/train_data_small_sparse.csv'
    vox_val_list = '/Lun0/zhiyong/sdsvc_small_100_dataset/source_vox_1211_smallval.csv'
    sdsvc_val_list = '/Lun0/zhiyong/sdsvc_small_100_dataset/target_sdsvc_100_smallval.csv'
    sdsvc_trial_list = '/Lun0/zhiyong/sdsvc_small_100_dataset/sdsvc_trial_list.csv'
    sdsvc_trial_keys = '/Lun0/zhiyong/sdsvc_small_100_dataset/sdsvc_trial_keys.csv'
    vox1test_trial_list = '/Lun0/zhiyong/sdsvc_small_100_dataset/vox1test_trial_list.csv'
    vox1test_trial_keys = '/Lun0/zhiyong/sdsvc_small_100_dataset/vox1test_trial_keys.csv'
    # vox1test_aux_list = '/Lun0/zhiyong/sdsvc_small_100_dataset/vox1test_aux_list.csv'
    # vox1test_aux_keys = '/Lun0/zhiyong/sdsvc_small_100_dataset/vox1test_aux_keys.csv'

    scoring_config = {'p_target': [0.01, 0.005], 'c_miss': 1, 'c_fa': 1}
    saver_metric = "MINC"
    lr_ctrl = {"metric": "MINC", "Dur": 4, "sig_th": 0.005}
    expect_scheduler_steps = 3

    val_interval_step = 500

    train_batch_size = 128

    gpu_id = "0, 1"
    num_workers = 32  # how many workers for loading data
    print_freq = 50  # print info every N batch

    lr = 1e-2  # initial learning rate

    weight_decay = 5e-4
    momentum = 0.9

    model_load_path = ''
    cudnn_benchmark = True

    exp_path = "/Lun2/rzz/kaldi-master/egs/zhiyong/sdsvc_exp/train_exp"
    tbx_path = "/Lun2/rzz/kaldi-master/egs/zhiyong/sdsvc_exp/tbx"
