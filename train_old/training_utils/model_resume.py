import os
import torch


def resume_training(opt, model, optimizer, scheduler):

    if opt.model_load_path == '':
        total_step = 0
        return model, optimizer, scheduler, total_step
    
    print("Restoring from "+opt.model_load_path)

    checkpoint = torch.load(opt.model_load_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=True)
    
    total_step = checkpoint['total_step']
    start_epoch = checkpoint['start_epoch']

    optimizer.load_state_dict(checkpoint['optimizer'])

    sch_statdict = scheduler.state_dict()
    sch_statdict['last_epoch'] = checkpoint['scheduler']['last_epoch'] - 1
    sch_statdict['_step_count'] = checkpoint['scheduler']['_step_count'] - 1

    print(sch_statdict)
    scheduler.load_state_dict(sch_statdict)
    scheduler.step()
  
    for i in range(99):
        if os.path.isdir(opt.train_name+'(re'+str(i)+')'):
            continue
        opt.train_name = opt.train_name+'(re'+str(i)+')'

    return model, optimizer, scheduler, total_step
