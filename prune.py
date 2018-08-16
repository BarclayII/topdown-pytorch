import argparse
from datasets import get_generator
from util import cuda
from modules import *
from constants import *

class PrunePolicy(object):
    def __init__(self, builder, readout, F_prepro=lambda x: x):
        builder.eval()
        readout.eval()
        self.builder = builder
        self.readout = readout
        self.builder.eval()
        self.readout.eval()
        self.F_prepro = F_prepro

    def predict(self, input):
        assert False, "Not implemented yet"

class PrunePolicyFrontier(PrunePolicy):
    def __init__(self, builder, readout, F_prepro=lambda x: x, policy=policy):
        """
        PrunePolicy: From root to current node.
        """
        super(PrunePolicyFrontier, self).__init__(builder, readout, F_prepro=F_prepro)
        self.policy = policy

    def predict(self, x):
        x_in = self.F_prepro(x)
        # TreeBuilder part
        lvl = self.builder.n_levels
        batch_size, n_channels, n_rows, n_cols = x.shape
        t = [TreeItem() for _ in range(num_nodes(lvl, self.builder.n_branches))]
        t[0].b = x.new(x.shape[0], self.builder.g_dims).zero_()

        for l in range(0, lvl + 1):
            current_level = self.builder.noderange(l)
            b = T.stack([t[i].b for i in current_level], 1)
            bbox, g, att, new_b, h = self.builder.forward_layer(x, l, b)
            for k, i in enumerate(current_level):
                t[i].bbox = bbox[:, k]
                t[i].g = g[:, k]
                t[i].h = h[:, k]
                t[i].att = att[:, k]
                if l != lvl:
                    for j in range(self.builder.n_branches):
                        t[i * self.builder.n_branches + j + 1].b = new_b[:, k, j]

        # Readout part
        result_ents = []
        n = num_nodes(lvl, self.readout.n_branches)
        nodes = [0]
        frontiers = list(self.builder.noderange(1))
        h_sub = T.stack([t[node].h for node in nodes], 1)
        ent_sub = F_ent(F.softmax(self.readout.predictor(h_sub.mean(dim=1)), dim=-1)).item()
        sel_node = 1
        sel_h = None
        while len(frontiers) and sel_node > 0:
            sel_node = 0
            for node in frontiers:
                h = T.cat([h_sub, t[node].h.unsqueeze(1)], 1)
                ent = F_ent(F.softmax(self.readout.predictor(h.mean(dim=1)), dim=-1)).item()
                if ent < ent_sub:
                    sel_node = node
                    ent_sub = ent
                    sel_h = h

            if sel_node > 0:
                frontiers.remove(sel_node)
                for j in range(self.builder.n_branches):
                    k = sel_node * self.builder.n_branches + j + 1
                    if k < len(t):
                        frontiers.append(k)
                nodes.append(sel_node)
                h_sub = sel_h
        return h_sub, ent_sub

if __name__ == '__main__':
    T.set_num_threads(4)
    parser = argparse.ArgumentParser(description='Prune')
    parser.add_argument('--resume', default=None, help='resume training from checkpoint')
    parser.add_argument('--row', default=200, type=int, help='image rows')
    parser.add_argument('--col', default=200, type=int, help='image cols')
    parser.add_argument('--n', default=100, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--log_interval', default=10, type=int, help='log interval')
    parser.add_argument('--share', action='store_true', help='indicates whether to share CNN params or not')
    parser.add_argument('--pretrain', action='store_true', help='pretrain or not pretrain')
    parser.add_argument('--hs', action='store_true', help='indicates whether to use homoscedastic uncertainty or not')
    parser.add_argument('--schedule', action='store_true', help='indicates whether to use schedule training or not')
    parser.add_argument('--att_type', default='mean', type=str, help='attention type: mean/naive/tanh')
    parser.add_argument('--pc_coef', default=1, type=float, help='regularization parameter(parent-child)')
    parser.add_argument('--cc_coef', default=0, type=float, help='regularization parameter(child-child)')
    parser.add_argument('--rank_coef', default=0.1, type=float, help='coefficient for rank loss')
    parser.add_argument('--branches', default=2, type=int, help='branches')
    parser.add_argument('--levels', default=2, type=int, help='levels')
    parser.add_argument('--rank', action='store_true', help='use rank loss')
    parser.add_argument('--backrand', default=0, type=int, help='background noise(randint between 0 to `backrand`)')
    parser.add_argument('--glm_type', default='gaussian', type=str, help='glimpse type (gaussian, bilinear)')
    parser.add_argument('--dataset', default='mnistmulti', type=str, help='dataset (mnistmulti, mnistcluttered, cifar10, imagenet)')
    parser.add_argument('--n_digits', default=1, type=int, help='indicate number of digits in multimnist dataset')
    parser.add_argument('--v_batch_size', default=1, type=int, help='valid batch size')
    parser.add_argument('--size_min', default=28 // 3 * 2, type=int, help='Object minimum size')
    parser.add_argument('--size_max', default=28, type=int, help='Object maximum size')
    parser.add_argument('--imagenet_root', default='/beegfs/qg323', type=str)
    parser.add_argument('--imagenet_train_sel', default='selected-train.pkl', type=str)
    parser.add_argument('--imagenet_valid_sel', default='selected-val.pkl', type=str)
    parser.add_argument('--num_workers', default=0, type=int)
    args = parser.parse_args()
    filter_arg_dict = {
            'resume': None,
            'v_batch_size': None,
            'batch_size': None,
            'n': None,
            'log_interval': None,
            'num_workers': None,
            'imagenet_root': None,
            'imagenet_train_sel': None,
            'imagenet_valid_sel': None
    }

    expr_setting = '_'.join('{}-{}'.format(k, v) for k, v in vars(args).items() if not k in filter_arg_dict)
    _, _, test_loader, preprocessor = get_generator(args)
    n_branches = args.branches
    n_levels = args.levels

    if args.dataset == 'imagenet':
        n_classes = 1000
        cnn = 'resnet18'
    elif args.dataset == 'cifar10':
        n_classes = 10
        cnn = None
    elif args.dataset.startswith('mnist'):
        n_classes = 10 ** args.n_digits
        cnn = None

    builder = cuda(TreeBuilder(n_branches=n_branches,
                        n_levels=n_levels,
                        att_type=args.att_type,
                        pc_coef=args.pc_coef,
                        cc_coef=args.cc_coef,
                        n_classes=n_classes,
                        glimpse_type=args.glm_type,
                        glimpse_size=GLIMPSE_SIZE,
                        cnn=cnn,))
    readout = cuda(ReadoutModule(n_branches=n_branches, n_levels=n_levels, n_classes=n_classes))
    batch_size = args.batch_size
    builder.load_state_dict(
        T.load('checkpoints/{}_builder_best.pt'.format(expr_setting)))
    readout.load_state_dict(
        T.load('checkpoints/{}_readout_best.pt'.format(expr_setting)))

    ppf = PrunePolicyFrontier(builder, readout)
    ent_mean = 0
    hit = 0
    from tqdm import tqdm
    for i, item in tqdm(enumerate(test_loader)):
        x, y, b = preprocessor(item)
        h, ent = ppf.predict(x)
        h = h.mean(dim=1)
        pred = ppf.readout.predictor(h).max(dim=-1)[1]
        hit += (pred == y).item()
        ent_mean += ent
#        lvl_ents = ppf.predict(x)
#        for j, ent_batch in enumerate(lvl_ents):
#            lvl_ent_mean[j] += ent_batch.mean().item()

    print(ent_mean / i, hit / i)
