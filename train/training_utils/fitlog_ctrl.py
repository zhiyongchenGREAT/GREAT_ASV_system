import fitlog
import os
import sys
import pickle
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../')
import score
from . import ckpt_saver 

__all__ = ['standard_fitlog_init', 'vox1test_ASV_st_fitlog']

def standard_fitlog_init(opt):
    fitlog.commit(file=opt.fitlog_file_dir, fit_msg=opt.train_name)             # auto commit your codes
    fitlog.set_log_dir(opt.fitlog_dir)         # set the logging directory
    
    fitlog.add_other({"DESCRIPTION": opt.description})
    fitlog.add_other({"DATASET": opt.fitlog_DATASET_log})

def vox1test_ASV_st_fitlog(model, opt, total_step, optimizer, scheduler, train_log):
    out_dir = os.path.join(opt.temporal_results_path, 's'+str(total_step), 'vox1test_ASV_eval')
    fitlog_bestlog = os.path.join(opt.temporal_results_path, 'fitlog_bestlog')
    fitlog_best_metric = opt.fitlog_best_metric

    fitlog_list = {"Voxceleb_O":{"EER":1.0, "MINC_0.01":1.0, "MINC_0.001":1.0}}
    if os.path.exists(fitlog_bestlog):
        with open(fitlog_bestlog, 'rb') as f:
            fitlog_best_list = pickle.load(f)
    else:
        fitlog_best_list = {"Voxceleb_O":{"EER":1.0, "MINC_0.01":1.0, "MINC_0.001":1.0}}

    scoring_config = {'p_target': [0.01], 'c_miss': 1, 'c_fa': 1}
    eer, minc_1, actc = score.scoring(os.path.join(out_dir, 'scores'), opt.vox1test_trial_keys, scoring_config)
    scoring_config = {'p_target': [0.001], 'c_miss': 1, 'c_fa': 1}
    _, minc_2, _ = score.scoring(os.path.join(out_dir, 'scores'), opt.vox1test_trial_keys, scoring_config)
    
    fitlog.add_metric({"Voxceleb_O":{"EER":eer}}, step=total_step)
    fitlog.add_metric({"Voxceleb_O":{"MINC_0.01":minc_1}}, step=total_step)
    fitlog.add_metric({"Voxceleb_O":{"MINC_0.001":minc_2}}, step=total_step)

    fitlog_list["Voxceleb_O"]["EER"] = eer
    fitlog_list["Voxceleb_O"]["MINC_0.01"] = minc_1
    fitlog_list["Voxceleb_O"]["MINC_0.001"] = minc_2

    if fitlog_list["Voxceleb_O"][fitlog_best_metric] <= fitlog_best_list["Voxceleb_O"][fitlog_best_metric]:
        with open(fitlog_bestlog, 'wb') as f:
            pickle.dump(fitlog_list, f)

        fitlog.add_best_metric({"Voxceleb_O":{"EER":eer*100}})
        fitlog.add_best_metric({"Voxceleb_O":{"MINC_0.01":minc_1}})
        fitlog.add_best_metric({"Voxceleb_O":{"MINC_0.001":minc_2}})

        msg = 'Fitlog add best Voxceleb_O '+fitlog_best_metric
        print(msg)
        train_log.writelines([msg+'\n']) 

        save_name = 'fitlog_best_Voxceleb_O_'+fitlog_best_metric
        save_log = 's'+str(total_step)+' '+'fitlog_best_Voxceleb_O_'+fitlog_best_metric+' '+str(fitlog_list["Voxceleb_O"][fitlog_best_metric])
        ckpt_saver.std_saver(model, opt, total_step, optimizer, scheduler, train_log, save_name, save_log)

