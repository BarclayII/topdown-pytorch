import torch.optim as optim

"""
TODO's:
1. check whether alpha(pytorch) is decay(tf)
2. minimum learning rate(in bird's case, min_lr=1e-4)
"""

OPTIM_PARAMS = {
    'mnistcluttered': {
        'mode': 'rmsprop',
        'args': {
            'lr': 1e-4,
            'weight_decay': 1e-5,
        },
        'scheduler_mode': 'none',
        'scheduler_args': {}
    },
    'mnistmulti': {
        'mode': 'rmsprop',
        'args': {
            'lr': 1e-4,
            'weight_decay': 1e-5,
        },
        'scheduler_mode': 'none',
        'scheduler_args': {}
    },
    'bird': {
        'mode': 'rmsprop',
        'args': {
            'lr': 3e-5,
            'weight_decay': 4e-5,
        },
        'scheduler_mode': 'steplr',
        'scheduler_args': {
            'step_size': 10,
            'gamma': 0.9
        }
    },
    'dogs': {
        'mode': 'sgd',
        'args': {
            'lr': 0.01,
            'weight_decay': 5e-5,
            'momentum': 0.9,
        },
        'scheduler_mode': 'steplr',
        'scheduler_args': {
            'step_size': 10,
            'gamma': 0.94
        }
    },
}

def create_optim(mode, params, args_dict):
    if mode == 'sgd':
        return optim.SGD(params, **args_dict)
    elif mode == 'adam':
        return optim.Adam(params, **args_dict)
    elif mode == 'rmsprop':
        return optim.RMSprop(params, **args_dict)
    else: # unexpected behavior
        raise KeyError('cannot find optimizer {}'.format(mode))
