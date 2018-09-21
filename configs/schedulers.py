from torch.optim.lr_scheduler import *

def create_scheduler(mode, optimizer, args_dict):
    if mode == 'none':
        return ExponentialLR(optimizer, 1.0)
    elif mode == 'steplr':
        return StepLR(optimizer, **args_dict)
