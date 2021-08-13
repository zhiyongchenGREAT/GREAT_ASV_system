#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Scheduler(optimizer, lr, base_lr, cycle_step, expected_step, **kwargs):

	sche_fn = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=lr, step_size_up=int(cycle_step//2), step_size_down=int(cycle_step//2), mode="triangular2", cycle_momentum=False)

	lr_step = 'iteration'

	print('Initialised Cyclic LR scheduler')

	return sche_fn, lr_step, expected_step
