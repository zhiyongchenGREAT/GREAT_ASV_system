def vox1test_cls_eval(model, opt, total_step, train_log, tbx_writer):
    
    val_data = PickleDataSet(opt.val_list)
    val_dataloader = My_DataLoader(val_data, batch_size=None, shuffle=False, sampler=None,\
    batch_sampler=None, num_workers=opt.num_workers, collate_fn=None,\
    pin_memory=False, drop_last=False, timeout=0,\
    worker_init_fn=None, multiprocessing_context=None)   
    
    model.eval()
    for val_count, (val_x, val_y) in enumerate(val_dataloader):

        val_x = val_x.cuda(non_blocking=True)
        val_y = val_y.cuda(non_blocking=True)
        
        with torch.no_grad():
            loss, predict, emb, acc, _ = model(val_x, val_y, mod='eval')
        
        val_loss += loss.item()
        val_acc += acc                     

    val_loss = val_loss / (val_count + 1)  
    val_acc = val_acc / (val_count + 1)

    writer.add_scalar('val/loss', val_loss, total_step)
    writer.add_scalar('val/acc', val_acc, total_step)
    writer_aux.add_scalar('val/loss', val_loss, total_step)
    writer_aux.add_scalar('val/acc', val_acc, total_step)



def vox1test_ASV_eval(model, opt, total_step, train_log, tbx_writer):