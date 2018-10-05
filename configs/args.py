import argparse

__all__ = ['args', 'expr_setting']

parser = argparse.ArgumentParser(description='Alternative')
parser.add_argument('--resume', default=None, help='resume training from checkpoint')
parser.add_argument('--row', default=200, type=int, help='image rows')
parser.add_argument('--col', default=200, type=int, help='image cols')
parser.add_argument('--n', default=100, type=int, help='number of epochs')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--log_interval', default=10, type=int, help='log interval')
parser.add_argument('--share', action='store_true', help='indicates whether to share CNN params or not')
parser.add_argument('--pretrain', action='store_true', help='pretrain or not pretrain')
parser.add_argument('--schedule', action='store_true', help='indicates whether to use schedule training or not')
parser.add_argument('--nearest', action='store_true', help='indicates whether to visualize nearest neighbors or not')
# PC coef is not necessary when using relative position. The coefficient is set to 0 by default.
parser.add_argument('--pc_coef', default=0, type=float, help='regularization parameter(parent-child)')
parser.add_argument('--cc_coef', default=0, type=float, help='regularization parameter(child-child)')
parser.add_argument('--res_coef', default=1, type=float, help='coefficient for resolution loss')
parser.add_argument('--branches', default=2, type=int, help='branches')
parser.add_argument('--levels', default=2, type=int, help='levels')
parser.add_argument('--levels_from', default=2, type=int, help='levels from')
parser.add_argument('--backrand', default=0, type=int, help='background noise(randint between 0 to `backrand`)')
parser.add_argument('--glm_type', default='gaussian', type=str, help='glimpse type (gaussian, bilinear)')
parser.add_argument('--dataset', default='mnistmulti', type=str, help='dataset (mnistmulti, mnistcluttered, cifar10, imagenet, flower, bird)')
parser.add_argument('--n_digits', default=1, type=int, help='indicate number of digits in multimnist dataset')
parser.add_argument('--v_batch_size', default=32, type=int, help='valid batch size')
parser.add_argument('--size_min', default=28 // 3 * 2, type=int, help='Object minimum size')
parser.add_argument('--size_max', default=28, type=int, help='Object maximum size')
parser.add_argument('--imagenet_root', default='/beegfs/qg323', type=str)
parser.add_argument('--imagenet_train_sel', default='selected-train.pkl', type=str)
parser.add_argument('--imagenet_valid_sel', default='selected-val.pkl', type=str)
parser.add_argument('--glm_size', default=12, type=int)
parser.add_argument('--explore', action='store_true', help='indicates whether to enable explore or not')
parser.add_argument('--bind', action='store_true', help='indicates whether to bind dx/sx, dy/sy or not')
parser.add_argument('--readout', default='maxgated', type=str)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--n_gpus', default=1, type=int)
parser.add_argument('--fix', action='store_true')
args = parser.parse_args()
filter_arg_dict = [
        'resume',
        'v_batch_size',
        'batch_size',
        'n',
        'log_interval',
        'imagenet_root',
        'imagenet_train_sel',
        'imagenet_valid_sel',
        'num_workers',
        'n_gpus',
]
expr_setting = '_'.join('{}'.format(v) for k, v in vars(args).items() if not k in filter_arg_dict)
