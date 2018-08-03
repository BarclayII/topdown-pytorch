import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as INIT
import torch.optim as optim
import numpy as np
import argparse
import os
import sys
from glimpse import create_glimpse
from util import cuda
from datasets import get_generator
from viz import fig_to_ndarray_tb
from tensorboardX import SummaryWriter
from stats.utils import *
from modules import *
import tqdm

T.set_num_threads(4)

parser = argparse.ArgumentParser(description='Alternative')
parser.add_argument('--resume', default=None, help='resume training from checkpoint')
parser.add_argument('--row', default=200, type=int, help='image rows')
parser.add_argument('--col', default=200, type=int, help='image cols')
parser.add_argument('--n', default=100, type=int, help='number of epochs')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--log_interval', default=10, type=int, help='log interval')
parser.add_argument('--share', action='store_true', help='indicates whether to share CNN params or not')
parser.add_argument('--pretrain', action='store_true', help='pretrain or not pretrain')
parser.add_argument('--schedule', action='store_true', help='indicates whether to use schedule training or not')
parser.add_argument('--att_type', default='mean', type=str, help='attention type: mean/naive/tanh')
parser.add_argument('--clip', default=0.1, type=float, help='gradient clipping norm')
parser.add_argument('--pc_coef', default=1, type=float, help='regularization parameter(parent-child)')
parser.add_argument('--cc_coef', default=1, type=float, help='regularization parameter(child-child)')
parser.add_argument('--rank_coef', default=0.1, type=float, help='coefficient for rank loss')
parser.add_argument('--branches', default=2, type=int, help='branches')
parser.add_argument('--levels', default=2, type=int, help='levels')
parser.add_argument('--rank', action='store_true', help='use rank loss')
parser.add_argument('--backrand', default=0, type=int, help='background noise(randint between 0 to `backrand`)')
parser.add_argument('--glm_type', default='gaussian', type=str, help='glimpse type (gaussian, bilinear)')
parser.add_argument('--dataset', default='mnistmulti', type=str, help='dataset (mnistmulti, cifar10)')
parser.add_argument('--v_batch_size', default=256, type=int, help='valid batch size')
parser.add_argument('--size_min', default=28 // 3 * 2, type=int, help='Object minimum size')
parser.add_argument('--size_max', default=28, type=int, help='Object maximum size')
args = parser.parse_args()
filter_arg_dict = {
        'resume': None,
        'v_batch_size': None,
        'batch_size': None,
        'n': None,
        'log_interval': None,
}
expr_setting = '_'.join('{}-{}'.format(k, v) for k, v in vars(args).items() if not k in filter_arg_dict)

data_generator, dataset_train, dataset_valid, train_sampler, valid_sampler = \
        get_generator(args)

writer = SummaryWriter('runs/{}'.format(expr_setting))

n_branches = args.branches
n_levels = args.levels

if args.resume is not None:
    builder = T.load('checkpoints/{}_builder_{}.pt'.format(expr_setting, args.resume))
    readout = T.load('checkpoints/{}_readout_{}.pt'.format(expr_setting, args.resume))
else:
    builder = cuda(TreeBuilder(n_branches=n_branches,
                            n_levels=n_levels,
                            att_type=args.att_type,
                            pc_coef=args.pc_coef,
                            cc_coef=args.cc_coef,
                            glimpse_type=args.glm_type))
    readout = cuda(ReadoutModule(n_branches=n_branches, n_levels=n_levels))

train_shuffle = True

n_epochs = args.n
batch_size = args.batch_size

len_train = len(dataset_train)
len_valid = len(dataset_valid)

loss_arr = []
acc_arr = []

def rank_loss(a, b, margin=0):
    #return (b - a + margin).clamp(min=0).mean()
    return F.sigmoid(b - a).mean()

def viz(epoch, imgs, bboxes, g_arr, att, tag):
    length = len(g_arr)
    statplot = StatPlot(5, 2)
    statplot_g_arr = [StatPlot(5, 2) for _ in range(length)]
    for j in range(10):
        statplot.add_image(
            imgs[j].permute(1, 2, 0),
            bboxs=[bbox[j] for bbox in bboxes],
            clrs=['y', 'y', 'r', 'r', 'r', 'r'],
            lws=att[j, 1:] * length
        )
        for k in range(length):
            statplot_g_arr[k].add_image(g_arr[k][j].permute(1, 2, 0), title='att_weight={}'.format(att[j, k]))
    writer.add_image('Image/{}/viz_bbox'.format(tag), fig_to_ndarray_tb(statplot.fig), epoch)
    for k in range(length):
        writer.add_image('Image/{}/viz_glim_{}'.format(tag, k), fig_to_ndarray_tb(statplot_g_arr[k].fig), epoch)
    plt.close('all')

def train():
    for epoch in range(n_epochs):
        print("Epoch {} starts...".format(epoch))
        params = list(builder.parameters()) + list(readout.parameters())
        opt = T.optim.RMSprop(params, lr=1e-4)

        start_lvl = 0 if args.schedule else n_levels
        train_loader = data_generator(dataset_train, batch_size, shuffle=True, sampler=train_sampler)
        sum_loss = 0
        n_batches = len_train // batch_size
        hit = 0
        cnt = 0
        levelwise_hit = np.zeros(n_levels + 1)

        for i, (x, y, b) in enumerate(train_loader):
            total_loss = 0

            t, loss_reg = builder(x)
            readout_list = readout(t)

			total_loss = loss_reg
            for lvl in range(start_lvl, n_levels + 1):
                y_pred, att_weights = readout_list[lvl]
                y_score = y_pred.gather(1, y[:, None])[:, 0]

                ce_loss = F.cross_entropy(y_pred, y)
                loss = ce_loss

                if args.rank and lvl > start_lvl:
                    loss_rank = rank_loss(y_score, y_score_last)
                    loss = loss + args.rank_coef * loss_rank
                y_score_last = y_score

                total_loss = total_loss + loss
                levelwise_hit[lvl] += (y_pred.max(dim=-1)[1] == y).sum().item()

            opt.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(params, args.clip)
            opt.step()
            sum_loss += total_loss.item()
            hit = levelwise_hit[-1]
            cnt += batch_size

            if i == 0:
                sample_imgs = x[:10]
                length = len(t)
                sample_bboxs = [glimpse_to_xyhw(t[k].bbox[:10, :4].detach() * args.row) for k in range(1, length)]
                sample_g_arr = [t[_].g[:10].detach() for _ in range(length)]
                sample_atts = att_weights.detach().cpu().numpy()[:10]

                viz(epoch, sample_imgs, sample_bboxs, sample_g_arr, sample_atts, 'train')

            if i % args.log_interval == 0 and i > 0:
                avg_loss = sum_loss / args.log_interval
                sum_loss = 0
                print('Batch {}/{}, loss = {}, acc = {}'.format(i, n_batches, avg_loss, hit * 1.0 / cnt))
                hit = 0
                cnt = 0
                levelwise_hit *= 0

        v_batch_size = args.v_batch_size
        valid_loader = data_generator(dataset_valid, v_batch_size, shuffle=False, sampler=valid_sampler)
        cnt = 0
        hit = 0
        levelwise_hit = np.zeros(n_levels + 1)
        sum_loss = 0
        with T.no_grad():
            for i, (x, y, b) in enumerate(valid_loader):
                total_loss = 0
                t, _ = builder(x)
                readout_list = readout(t)

                for lvl in range(start_lvl, n_levels + 1):
                    y_pred, att_weights = readout_list[lvl]
                    loss = F.cross_entropy(
                        y_pred, y
                    )
                    total_loss += loss
                    levelwise_hit[lvl] += (y_pred.max(dim=-1)[1] == y).sum().item()

                sum_loss += total_loss.item()
                hit = levelwise_hit[-1]
                cnt += v_batch_size

                if i == 0:
                    sample_imgs = x[:10]
                    length = len(t)
                    sample_bboxs = [glimpse_to_xyhw(t[k].bbox[:10, :4] * args.row) for k in range(1, length)]
                    sample_g_arr = [t[_].g[:10] for _ in range(length)]
                    sample_atts = att_weights.cpu().numpy()[:10]

                    viz(epoch, sample_imgs, sample_bboxs, sample_g_arr, sample_atts, 'valid')

        avg_loss = sum_loss / i
        acc = hit * 1.0 / cnt
        levelwise_acc = levelwise_hit * 1.0 / cnt
        print("Loss on valid set: {}".format(avg_loss))
        print("Accuracy on valid set: {}".format(acc))

        writer.add_scalar('data/loss', avg_loss, epoch)
        if args.schedule:
            writer.add_scalars('data/levelwise_loss', {str(lvl): levelwise_acc[lvl] for lvl in range(n_levels + 1)}, epoch)
        writer.add_scalar('data/accuracy', acc, epoch)

        if (epoch + 1) % 10 == 0:
            print('Save checkpoint...')
            if not os.path.exists('checkpoints/'):
                os.makedirs('checkpoints')
            T.save(builder, 'checkpoints/{}_builder_{}.pt'.format(expr_setting, epoch))
            T.save(readout, 'checkpoints/{}_readout_{}.pt'.format(expr_setting, epoch))

if __name__ == '__main__':
    train()
