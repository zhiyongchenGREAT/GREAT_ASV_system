#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Scheduler(optimizer, lr, base_lr, expected_step, **kwargs):

	sche_fn = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, expected_step, T_mult=1, eta_min=base_lr, last_epoch=-1)
	
	lr_step = 'iteration'

	print('Initialised CosineAnnealingWarmRestarts LR scheduler')

	return sche_fn, lr_step, expected_step
