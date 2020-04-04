
def resume_training(opt, model, optimizer, scheduler):
    print("Restoring from "+opt.model_load_path)
    total_step = checkpoint['total_step']
    start_epoch = checkpoint['start_epoch']

    optimizer.load_state_dict(checkpoint['optimizer'])

    sch_statdict = scheduler.state_dict()
    sch_statdict['last_epoch'] = checkpoint['scheduler']['last_epoch'] - 1
    sch_statdict['_step_count'] = checkpoint['scheduler']['_step_count'] - 1
    logger.info(sch_statdict)
    scheduler.load_state_dict(sch_statdict)
    scheduler.step()

    checkpoint = torch.load(opt.model_load_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=True)

    return model, optimizer, scheduler
