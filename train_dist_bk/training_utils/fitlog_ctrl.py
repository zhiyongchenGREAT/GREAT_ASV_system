import fitlog
import os
import sys
import pickle


__all__ = ['standard_fitlog_init', 'vox1_o_ASV_step_fitlog', 'vox1_o_ASV_best_fitlog', \
    'sdsvdev_ASV_step_fitlog', 'sdsvdev_ASV_best_fitlog']

def standard_fitlog_init(fitlogdir, train_name, fitlog_DATASET, fitlog_Desc, **kwargs):
    fitlog.commit(__file__, fit_msg=train_name)             # auto commit your codes
    fitlog.set_log_dir(fitlogdir)         # set the logging directory
    
    fitlog.add_other({"DESCRIPTION": fitlog_Desc})
    fitlog.add_other({"DATASET": fitlog_DATASET})

def vox1_o_ASV_step_fitlog(eer, minc_1, minc_2, step):
    fitlog.add_metric({"Voxceleb_O":{"EER":eer}}, step=step)
    fitlog.add_metric({"Voxceleb_O":{"MINC_0.01":minc_1}}, step=step)
    fitlog.add_metric({"Voxceleb_O":{"MINC_0.001":minc_2}}, step=step)

def vox1_o_ASV_best_fitlog(eer, minc_1, minc_2):
    fitlog.add_best_metric({"Voxceleb_O":{"EER":eer}})
    fitlog.add_best_metric({"Voxceleb_O":{"MINC_0.01":minc_1}})
    fitlog.add_best_metric({"Voxceleb_O":{"MINC_0.001":minc_2}})

def sdsvdev_ASV_step_fitlog(eer, minc_1, minc_2, step):
    fitlog.add_metric({"SDSV20_DEV":{"EER":eer}}, step=step)
    fitlog.add_metric({"SDSV20_DEV":{"MINC_0.01":minc_1}}, step=step)
    fitlog.add_metric({"SDSV20_DEV":{"MINC_0.001":minc_2}}, step=step)

def sdsvdev_ASV_best_fitlog(eer, minc_1, minc_2):
    fitlog.add_best_metric({"SDSV20_DEV":{"EER":eer}})
    fitlog.add_best_metric({"SDSV20_DEV":{"MINC_0.01":minc_1}})
    fitlog.add_best_metric({"SDSV20_DEV":{"MINC_0.001":minc_2}})
