class ExpConfig(object):
    def __init__(self, dataset='cluttered'):
        if dataset == 'cluttered':
            self.full_fm_shape = (16, 16)
            self.glimpse_fm_shape = (4, 4)
            self.h_dims = 512
        elif dataset == 'bird':
            self.full_fm_shape = (9, 9)
            self.glimpse_fm_shape = (2, 2)
            self.h_dims = 128
        else: # default
            assert False, "Not specified yet"
            pass
