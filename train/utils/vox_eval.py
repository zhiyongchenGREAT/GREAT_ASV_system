def vox1test_cls_eval(model, opt, total_step, train_log, tbx_writer):

    val_data = PickleDataSet(opt.vox_val_list)
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

    tbx_writer.add_scalar('vox1test_cls_eval_loss', val_loss, total_step)
    tbx_writer.add_scalar('vox1test_cls_eval_acc', val_acc, total_step)

    msg = "vox1test_cls_eval Step: {:} \
    ValLoss: {:.4f} ValAcc: {:.4f}".format(total_step, val_loss, val_acc)
    print(msg)
    train_log.writelines([msg+'\n']) 
    with open(opt.val_log_path, 'a') as f:
        f.writelines([msg+'\n']) 


def vox1test_ASV_eval(model, opt, total_step, train_log, tbx_writer):
    test_list = {}

    torch.backends.cudnn.benchmark = False
    model.eval()
    print('Final score evaluation')

    train_data = PickleDataSet(opt.vox1test_trial_list)
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
        # if (count+1) % 500 == 0:
        #     print(count+1)

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