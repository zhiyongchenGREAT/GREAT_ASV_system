def sdsvc_cls_eval(model, opt, total_step, train_log, tbx_writer):

    val_data = PickleDataSet(opt.sdsvc_val_list)
    val_dataloader = My_DataLoader(val_data, batch_size=None, shuffle=False, sampler=None,\
    batch_sampler=None, num_workers=opt.num_workers, collate_fn=None,\
    pin_memory=False, drop_last=False, timeout=0,\
    worker_init_fn=None, multiprocessing_context=None)   
    
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    for val_count, (val_x, val_y) in enumerate(val_dataloader):

        val_x = val_x.cuda(non_blocking=True)
        val_y = val_y.cuda(non_blocking=True)
        
        with torch.no_grad():
            loss, predict, emb, acc, _ = model(val_x, val_y, mod='eval')
        
        val_loss += loss.item()
        val_acc += acc                     

    val_loss = val_loss / (val_count + 1)  
    val_acc = val_acc / (val_count + 1)

    tbx_writer.add_scalar('sdsvc_cls_eval_loss', val_loss, total_step)
    tbx_writer.add_scalar('sdsvc_cls_eval_acc', val_acc, total_step)

    msg = "sdsvc_cls_eval Step: {:} Valcount: {:} \
    ValLoss: {:.4f} ValAcc: {:.4f}".format(total_step, (val_count + 1), val_loss, val_acc)
    print(msg)
    train_log.writelines([msg+'\n']) 
    with open(opt.val_log_path, 'a') as f:
        f.writelines([msg+'\n']) 


def sdsvc_ASV_eval(model, opt, total_step, train_log, tbx_writer):
    torch.backends.cudnn.benchmark = False
    model.eval()
    # print('Final score evaluation')

    train_data = PickleDataSet(opt.sdsvc_trial_list)
    train_dataloader = My_DataLoader(train_data, batch_size=None, shuffle=False, sampler=None,\
    batch_sampler=None, num_workers=opt.num_workers, collate_fn=None,\
    pin_memory=False, drop_last=False, timeout=0,\
    worker_init_fn=None, multiprocessing_context=None)

    test_list = {}

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
        # if (count+1) % 500 == 0:
        #     print(count+1)

    msg = "sdsvc_ASV_eval Step: {:} Embcount: {:}".format(total_step, (count + 1))
    print(msg)
    train_log.writelines([msg+'\n']) 
    
    out_dir = os.path.join(opt.temporal_results_path, 's'+str(total_step), 'sdsvc_ASV_eval')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    f_out = open(os.path.join(out_dir, 'scores'), 'w')   
    
    for i in test_list:
        test_list[i] = (1.0 / np.linalg.norm(test_list[i])) * test_list[i]
    
    with open(opt.vox1test_trial_keys, 'r') as f:
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

    msg = "sdsvc_trial_keys Step: {:} Trialcount: {:}".format(total_step, (count + 1))
    print(msg)
    train_log.writelines([msg+'\n']) 

    if hasattr(opt, 'sdsvc_aux_list') and hasattr(opt, 'sdsvc_aux_keys'):
        train_data = PickleDataSet(opt.sdsvc_aux_list)
        train_dataloader = My_DataLoader(train_data, batch_size=None, shuffle=False, sampler=None,\
        batch_sampler=None, num_workers=opt.num_workers, collate_fn=None,\
        pin_memory=False, drop_last=False, timeout=0,\
        worker_init_fn=None, multiprocessing_context=None)

        test_list = {}

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
            # if (count+1) % 500 == 0:
            #     print(count+1)

        msg = "sdsvc_ASV_eval Step: {:} AuxEmbcount: {:}".format(total_step, (count + 1))
        print(msg)
        train_log.writelines([msg+'\n']) 
        
        out_dir = os.path.join(opt.temporal_results_path, 's'+str(total_step), 'sdsvc_ASV_eval')
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        f_out = open(os.path.join(out_dir, 'scores_aux'), 'w')   
        
        for i in test_list:
            test_list[i] = (1.0 / np.linalg.norm(test_list[i])) * test_list[i]
        
        with open(opt.vox1test_trial_keys, 'r') as f:
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

        msg = "sdsvc_ASV_eval Step: {:} TrialAuxcount: {:}".format(total_step, (count + 1))
        print(msg)
        train_log.writelines([msg+'\n'])

    if hasattr(opt, 'sdsvc_aux_list') and hasattr(opt, 'sdsvc_aux_keys'):
        calibrating(os.path.join(out_dir, 'calib.pth'), 50, opt.sdsvc_aux_keys, [os.path.join(out_dir, 'scores_aux')])
        applying(os.path.join(out_dir, 'calib.pth'), [os.path.join(out_dir, 'scores')], os.path.join(out_dir, 'scores_calib'))
        eer, minc, actc = scoring(os.path.join(out_dir, 'scores_calib'), opt.sdsvc_trial_keys, opt.scoring_config)
    else:
        eer, minc, actc = scoring(os.path.join(out_dir, 'scores'), opt.sdsvc_trial_keys, opt.scoring_config)

    tbx_writer.add_scalar('sdsvc_ASV_eval_EER', eer, total_step)
    tbx_writer.add_scalar('sdsvc_ASV_eval_MINC', minc, total_step)

    msg = "sdsvc_ASV_eval Step: {:} EER: {:.4f} \
    MINC: {:.4f} ACTC: {:.4f}".format(total_step, eer, minc, actc)
    print(msg)
    train_log.writelines([msg+'\n']) 
    with open(opt.val_log_path, 'a') as f:
        f.writelines([msg+'\n'])    
        