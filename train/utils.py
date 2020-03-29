import torch
import numpy as np
import os
from read_data import *
import numpy as np

import sre_scorer as sc
from my_dataloader import My_DataLoader
from my_scorer import scoring
from calibrate_scores import calibrating
from apply_calibration import applying

def score_me(score_list, label_list, configuration):
    scores = score_list
    tar_nontar_labs = label_list
    
    results = {}

    p_target = configuration['p_target']
    c_miss = configuration['c_miss']
    c_fa = configuration['c_fa']
    act_c_avg = 0.
    for p_t in p_target:
        act_c, _, _ = sc.compute_actual_cost(scores,\
        tar_nontar_labs, p_t,\
        c_miss, c_fa)
        act_c_avg += act_c
    act_c_avg = act_c_avg / len(p_target)

    weights = None
    fnr, fpr = sc.compute_pmiss_pfa_rbst(scores, tar_nontar_labs, weights)
    eer = sc.compute_eer(fnr, fpr)
    avg_min_c = 0.
    for p_t in p_target:
        avg_min_c += sc.compute_c_norm(fnr, fpr, p_t)
    avg_min_c = avg_min_c / len(p_target)
    
    results['OUT'] = [eer, avg_min_c, act_c_avg]
    return results

def trial_eval(model, opt, device, out_dir=None):
    # Evaluation
    test_list = {}

    torch.backends.cudnn.benchmark = False
    model.eval()
    print('Final score evaluation')

    # train_data = CSVDataSet(opt.trial_list)
    # train_dataloader = DataLoader(dataset=train_data, batch_size = 1, shuffle = False, num_workers = opt.num_workers, pin_memory=True)

    train_data = PickleDataSet(opt.trial_list)
    train_dataloader = My_DataLoader(train_data, batch_size=None, shuffle=False, sampler=None,\
    batch_sampler=None, num_workers=opt.num_workers, collate_fn=None,\
    pin_memory=False, drop_last=False, timeout=0,\
    worker_init_fn=None, multiprocessing_context=None)

    for count, (batch_x, batch_y) in enumerate(train_dataloader):
        batch_x = batch_x.to(device)
        label = batch_y[0]

        batch_y = torch.tensor([0]).to(device)
        
        with torch.no_grad():
            _, _, emb, _, _ = model(batch_x, batch_y, mod='eval')

        emb = emb.squeeze().data.cpu().numpy()
        
        if label not in test_list.keys():
            test_list[label] = emb[None, :]
        else:
            print('repeat eer:', label)
            break
    #         test_list[label] = np.append(test_list[label], emb[None, :], axis=0)
        
        if (count+1) % 500 == 0:
            print(count+1)

    if out_dir is None:
        out_dir = opt.final_results_path
        f_out = open(os.path.join(out_dir, 'scores'), 'w')
    else:
        f_out = open(os.path.join(out_dir, 'scores'), 'w')
    
    
    for i in test_list:
        test_list[i] = (1.0 / np.linalg.norm(test_list[i])) * test_list[i]
    
    with open(opt.trial_path, 'r') as f:
        for count, line in enumerate(f):
            if count == 0:
                print(line)

            enroll_emb = test_list[line.split(' ')[0][:-4]].squeeze()
            test_emb = test_list[line.split(' ')[1][:-4]].squeeze()
            
            cosine = np.dot(enroll_emb, test_emb)
            
            f_out.write(line.split(' ')[0]+' '+line.split(' ')[1]+' '+str(cosine)+'\n')
            
            if (count+1) % 5000 == 0:
                print(count+1)
    
    f_out.close()

    if opt.score_calib:
        # _ = scoring([os.path.join(out_dir, 'scores')], opt.trial_path)
        calibrating(os.path.join(out_dir, 'calib.pth'), 50, opt.trial_path, [os.path.join(out_dir, 'scores')])
        applying(os.path.join(out_dir, 'calib.pth'), [os.path.join(out_dir, 'scores')], os.path.join(out_dir, 'scores_calib'))
        results = scoring(os.path.join(out_dir, 'scores_calib'), opt.trial_path)

    else:
        results = scoring(os.path.join(out_dir, 'scores'), opt.trial_path)
    
    for ds, res in results.items():
        eer, minc, actc = res

    # score_list = []
    # with open(os.path.join(out_dir, 'scores'), 'r') as f:
    #     for i in f:
    #         score_list.append(float(i.split(' ')[-1][:-1]))
    # score_list = np.array(score_list, dtype=np.float)
    
    # label_list = []
    # for i in range(len(score_list)):
    #     label_list.append((i+1) % 2)
    # label_list = np.array(label_list, dtype=np.int)

    # configuration = {'p_target': [0.01, 0.005], 'c_miss': 1, 'c_fa': 1}
    # results = score_me(score_list, label_list, configuration)
    # print('\nSet\tEER[%]\tmin_C\tact_C')
    # for ds, res in results.items():
    #     eer, minc, actc = res
    #     print('{}\t{:05.2f}\t{:.3f}\t{:.3f}'.format(ds.upper(), eer*100,
    #           minc, actc))
    
    return eer, minc, actc

def trial_eval_2(model, opt, device, out_dir=None):
    # Evaluation
    test_list = {}

    torch.backends.cudnn.benchmark = False
    model.eval()
    print('Final score evaluation')

    # train_data = CSVDataSet(opt.trial_list)
    # train_dataloader = DataLoader(dataset=train_data, batch_size = 1, shuffle = False, num_workers = opt.num_workers, pin_memory=True)

    train_data = PickleDataSet(opt.trial_list_2)
    train_dataloader = My_DataLoader(train_data, batch_size=None, shuffle=False, sampler=None,\
    batch_sampler=None, num_workers=opt.num_workers, collate_fn=None,\
    pin_memory=False, drop_last=False, timeout=0,\
    worker_init_fn=None, multiprocessing_context=None)

    for count, (batch_x, batch_y) in enumerate(train_dataloader):
        batch_x = batch_x.to(device)
        label = batch_y[0]

        batch_y = torch.tensor([0]).to(device)
        
        with torch.no_grad():
            _, _, emb, _, _ = model(batch_x, batch_y, mod='eval')

        emb = emb.squeeze().data.cpu().numpy()
        
        if label not in test_list.keys():
            test_list[label] = emb[None, :]
        else:
            print('repeat eer:', label)
            break
    #         test_list[label] = np.append(test_list[label], emb[None, :], axis=0)
        
        if (count+1) % 500 == 0:
            print(count+1)

    if out_dir is None:
        out_dir = opt.final_results_path
        f_out = open(os.path.join(out_dir, 'scores'), 'w')
    else:
        f_out = open(os.path.join(out_dir, 'scores'), 'w')
    
    
    for i in test_list:
        test_list[i] = (1.0 / np.linalg.norm(test_list[i])) * test_list[i]
    
    with open(opt.trial_path_2, 'r') as f:
        for count, line in enumerate(f):
            if count == 0:
                print(line)

            enroll_emb = test_list[line.split(' ')[0][:-4]].squeeze()
            test_emb = test_list[line.split(' ')[1][:-4]].squeeze()
            
            cosine = np.dot(enroll_emb, test_emb)
            
            f_out.write(line.split(' ')[0]+' '+line.split(' ')[1]+' '+str(cosine)+'\n')
            
            if (count+1) % 5000 == 0:
                print(count+1)
    
    f_out.close()

    if opt.score_calib:
        # _ = scoring([os.path.join(out_dir, 'scores')], opt.trial_path_2)
        calibrating(os.path.join(out_dir, 'calib.pth'), 50, opt.trial_path_2, [os.path.join(out_dir, 'scores')])
        applying(os.path.join(out_dir, 'calib.pth'), [os.path.join(out_dir, 'scores')], os.path.join(out_dir, 'scores_calib'))
        results = scoring(os.path.join(out_dir, 'scores_calib'), opt.trial_path_2)

    else:
        results = scoring(os.path.join(out_dir, 'scores'), opt.trial_path_2)
    
    for ds, res in results.items():
        eer, minc, actc = res

    # score_list = []
    # with open(os.path.join(out_dir, 'scores'), 'r') as f:
    #     for i in f:
    #         score_list.append(float(i.split(' ')[-1][:-1]))
    # score_list = np.array(score_list, dtype=np.float)
    
    # label_list = []
    # for i in range(len(score_list)):
    #     label_list.append((i+1) % 2)
    # label_list = np.array(label_list, dtype=np.int)

    # configuration = {'p_target': [0.01, 0.005], 'c_miss': 1, 'c_fa': 1}
    # results = score_me(score_list, label_list, configuration)
    # print('\nSet\tEER[%]\tmin_C\tact_C')
    # for ds, res in results.items():
    #     eer, minc, actc = res
    #     print('{}\t{:05.2f}\t{:.3f}\t{:.3f}'.format(ds.upper(), eer*100,
    #           minc, actc))
    
    return eer, minc, actc