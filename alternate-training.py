import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as INIT
import torch.optim as optim
from torchvision.models import resnet18
import numpy as np
import argparse
import os
import sys
from glimpse import create_glimpse
from util import cuda
from datasets import get_generator
from viz import fig_to_ndarray_tb
from tensorboardX import SummaryWriter
from stats import *
from modules import *
import tqdm

T.set_num_threads(4)
temp_arr = [1, 0.2, 0.01]

parser = argparse.ArgumentParser(description='Alternative')
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
parser.add_argument('--l2_coef', default=1, type=float, help='coefficient for l2 loss')
parser.add_argument('--branches', default=2, type=int, help='branches')
parser.add_argument('--levels', default=2, type=int, help='levels')
parser.add_argument('--levels_from', default=2, type=int, help='levels from')
parser.add_argument('--rank', action='store_true', help='use rank loss')
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
parser.add_argument('--glm_size', default=15, type=int)
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

train_loader, valid_loader, test_loader, preprocessor = get_generator(args)

writer = SummaryWriter('runs/{}'.format(expr_setting))

n_branches = args.branches
n_levels = args.levels
if args.dataset == 'imagenet':
    n_classes = 1000
    cnn = 'resnet18'
    in_dims = None
elif args.dataset == 'dogs':
    n_classes = 120
    cnn = 'resnet50'
    in_dims = None
elif args.dataset == 'cifar10':
    n_classes = 10
    from pytorch_cifar.models import ResNet18
    cnn = ResNet18()
    cnn.load_state_dict(T.load('cnntest.pt'))
    cnn = nn.Sequential(
            cnn.conv1,
            cnn.bn1,
            nn.ReLU(),
            cnn.layer1,
            cnn.layer2,
            cnn.layer3,
            cnn.layer4,
            nn.AdaptiveAvgPool2d(1),
            )
    in_dims = 512
elif args.dataset.startswith('mnist'):
    n_classes = 10 ** args.n_digits
    cnn = None
    in_dims = None
elif args.dataset == 'flower':
    n_classes = 102
    cnn = 'resnet18'
    in_dims = None
elif args.dataset == 'bird':
    n_classes = 200
    cnn = 'resnet18'
    in_dims = None

if args.hs is True:
    """
    0: Cross-Entropy Loss
    1: Rank Loss
    2: PC Loss
    3: CC Loss
    4: L2 Loss
    """
    hs = cuda(HomoscedasticModule())
    args.pc_coef = 1
    args.cc_coef = 0
    args.rank_coef = 1

if args.resume is not None:
    builder = T.load('checkpoints/{}_builder_{}.pt'.format(expr_setting, args.resume))
    readout = T.load('checkpoints/{}_readout_{}.pt'.format(expr_setting, args.resume))
else:
    builder = cuda(nn.DataParallel(TreeBuilder(n_branches=n_branches,
                            n_levels=n_levels,
                            att_type=args.att_type,
                            pc_coef=args.pc_coef,
                            cc_coef=args.cc_coef,
                            l2_coef=args.l2_coef,
                            n_classes=n_classes,
                            glimpse_type=args.glm_type,
                            glimpse_size=(args.glm_size, args.glm_size),
                            what__cnn=cnn,
                            what__fix=args.fix,
                            what__in_dims=in_dims)))
    readout = cuda(nn.DataParallel(ReadoutModule(n_branches=n_branches, n_levels=n_levels, n_classes=n_classes)))

train_shuffle = True

n_epochs = args.n
batch_size = args.batch_size

loss_arr = []
acc_arr = []

def rank_loss(a, b, margin=0):
    #return (b - a + margin).clamp(min=0).mean()
    return F.sigmoid(b - a).mean()

def imagenet_normalize(x):
    mean = T.FloatTensor([0.485, 0.456, 0.406]).to(x)
    std = T.FloatTensor([0.229, 0.224, 0.225]).to(x)
    x = (x - mean[None, :, None, None]) / std[None, :, None, None]
    return x

available_clrs = ['y', 'r', 'g', 'b']
def viz(epoch, imgs, bboxes, g_arr, att, tag, n_branches=2, n_levels=2):
    length = len(g_arr)
    statplot = StatPlot(5, 2)
    statplot_g_arr = [StatPlot(5, 2) for _ in range(length)]

    clrs = []
    for j in range(n_levels):
        clrs += [available_clrs[j]] * (n_branches ** (j + 1))
    for j in range(10):
        statplot.add_image(
            imgs[j].permute(1, 2, 0),
            bboxs=[bbox[j] for bbox in bboxes],
            clrs=clrs, #['y', 'y', 'r', 'r', 'r', 'r'],
            lws=att[j, 1:] * length
        )
        for k in range(length):
            statplot_g_arr[k].add_image(g_arr[k][j].permute(1, 2, 0), title='att_weight={}'.format(att[j, k]))

    statplot_disp_g = StatPlot(5, 2)
    channel, row, col = imgs[-1].shape
    for j in range(10):
        bbox_list = [
            np.array([0, 0, col, row])
        ] + [
            bbox_batch[j] for bbox_batch in bboxes
        ]
        glim_list = [
            g_arr[k][j].permute(1, 2, 0) for k in range(length)]
        statplot_disp_g.add_image(
            display_glimpse(channel, row, col, bbox_list, glim_list))
    writer.add_image('Image/{}/disp_glim'.format(tag), fig_to_ndarray_tb(statplot_disp_g.fig), epoch)
    writer.add_image('Image/{}/viz_bbox'.format(tag), fig_to_ndarray_tb(statplot.fig), epoch)
    for k in range(length):
        writer.add_image('Image/{}/viz_glim_{}'.format(tag, k), fig_to_ndarray_tb(statplot_g_arr[k].fig), epoch)
    plt.close('all')

logfile = open('debug.log', 'w')
dataset_with_sgd_schedule = ['cifar10', 'dogs']
dataset_with_normalize = ['cifar10', 'imagenet', 'flower', 'dogs', 'bird']

#@profile
def train():
    best_epoch = 0
    best_valid_loss = 1e6
    best_acc = 0
    levels = args.levels_from

    n_train_batches = len(train_loader)

    params = list(builder.parameters()) + list(readout.parameters()) + (list(hs.parameters()) if args.hs else [])
    if args.dataset.startswith('mnist'):
        lr = 1e-4
        opt = T.optim.RMSprop(params, lr=1e-4)
    elif args.dataset in dataset_with_sgd_schedule:
        lr = 0.01
        opt = T.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-4)
        #opt = T.optim.RMSprop(params, lr=1e-4, weight_decay=5e-4)
    elif args.dataset in ['imagenet', 'dogs']:
        lr = 0.1
        opt = T.optim.RMSprop(params, lr=3e-5, weight_decay=1e-4)
    elif args.dataset == 'flower' or args.dataset == 'bird':
        lr = 1e-4
        opt = T.optim.RMSprop(params, lr=1e-4)

    for epoch in range(n_epochs):
        print("Epoch {} starts...".format(epoch))

        readout_start_lvl = 0 #levels
        sum_loss = 0
        train_loss_dict = {
            'pc': 0.,
            'cc': 0.,
            'rank': 0,
            'ce': 0,
            'l2': 0,
        }
        hit = 0
        cnt = 0
        levelwise_hit = np.zeros(n_levels + 1)

        builder.train(mode=True)
        readout.train(mode=True)
        # I can directly use tqdm.tqdm(train_loader) but it doesn't display progress bar and time estimates
        with tqdm.tqdm(train_loader) as tq:
            #x, y, b = preprocessor(next(train_iter))
            for i, item in enumerate(tq):
                x, y, b = preprocessor(item)
                print(x.shape, y.shape, file=logfile)

                if args.dataset in dataset_with_normalize:
                    x_in = imagenet_normalize(x)
                else:
                    x_in = x

                total_loss = 0

                t, (loss_pc, loss_cc, loss_l2) = builder(x_in, levels)
                loss_pc = loss_pc.mean()
                loss_cc = loss_cc.mean()
                loss_l2 = loss_l2.mean()
                train_loss_dict['pc'] += loss_pc.item()
                train_loss_dict['cc'] += loss_cc.item()
                train_loss_dict['l2'] += loss_l2.item()
                readout_list = readout(t, levels)

                if args.hs:
                    total_loss =\
                        loss_pc * hs.coef_lambda[2] +\
                        loss_cc * hs.coef_lambda[3]
                else:
                    total_loss = loss_pc + loss_cc + loss_l2
                for lvl in range(readout_start_lvl, levels + 1):
                    y_pred, att_weights = readout_list[lvl]
                    y_score = y_pred.gather(1, y[:, None])[:, 0]

                    loss_ce = kl_temperature(y_pred, y, temperature=temp_arr[lvl]) #F.cross_entropy(y_pred, y)
                    train_loss_dict['ce'] += loss_ce.item()
                    if args.hs:
                        loss = loss_ce * hs.coef_lambda[0]
                    else:
                        loss = loss_ce

                    if args.rank and lvl > readout_start_lvl:
                        loss_rank = rank_loss(y_score, y_score_last)
                        if args.hs:
                            loss = loss + loss_rank * hs.coef_lambda[1]
                        else:
                            loss = loss + args.rank_coef * loss_rank
                        train_loss_dict['rank'] += args.rank_coef * loss_rank.item()

                    y_score_last = y_score

                    total_loss = total_loss + loss
                    if args.hs:
                        """
                        Homoscedastic Loss
                        """
                        total_loss = total_loss - (T.log(hs.coef_lambda)).sum()

                    current_hit = (y_pred.max(dim=-1)[1] == y).sum().item()
                    levelwise_hit[lvl] += current_hit

                opt.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(params, 0.1)
                opt.step()
                sum_loss += total_loss.item()
                hit = levelwise_hit[levels] # - 1]
                cnt += batch_size

                if i == 0:
                    sample_imgs = x[:10]
                    bbox_scaler = T.FloatTensor([[x.shape[3], x.shape[2], x.shape[3], x.shape[2]]]).to(x)
                    length = len(t)
                    sample_bboxs = [
                            glimpse_to_xyhw(t[k].bbox[:10, :4].detach()) * bbox_scaler
                            for k in range(1, length)
                            ]
                    sample_g_arr = [t[_].g[:10].detach() for _ in range(length)]
                    sample_atts = att_weights.detach().cpu().numpy()[:10]

                    viz(epoch, sample_imgs, sample_bboxs, sample_g_arr, sample_atts, 'train', n_branches=n_branches, n_levels=n_levels)

                tq.set_postfix({
                    'train_loss': total_loss.item(),
                    'train_acc': current_hit / batch_size,
                    })
                if i == n_train_batches - 1:
                    tq.set_postfix({
                        'train_avg_loss': sum_loss / n_train_batches,
                        'train_avg_acc': hit / cnt,
                        })
        for key in train_loss_dict.keys():
            train_loss_dict[key] /= n_train_batches
        writer.add_scalars('data/train_loss_dict', train_loss_dict, epoch)
        if args.hs:
            coef_lambda = hs.coef_lambda.cpu().tolist()
            lambda_dict = {
                'ce': coef_lambda[0],
                'rank': coef_lambda[1],
                'pc': coef_lambda[2],
                'cc': coef_lambda[3]
            }
            writer.add_scalars('data/coef_lambda', lambda_dict, epoch)

        cnt = 0
        hit = 0
        levelwise_hit = np.zeros(n_levels + 1)
        sum_loss = 0
        builder.eval()
        readout.eval()
        with T.no_grad():
            for i, item in enumerate(tqdm.tqdm(valid_loader)):
                x, y, b = preprocessor(item)
                print(x.shape, y.shape, file=logfile)
                if args.dataset in dataset_with_normalize:
                    x_in = imagenet_normalize(x)
                else:
                    x_in = x

                total_loss = 0
                t, _ = builder(x_in, levels)
                readout_list = readout(t, levels)

                for lvl in range(readout_start_lvl, levels + 1):
                    y_pred, att_weights = readout_list[lvl]
                    loss = kl_temperature(
                        y_pred, y, temperature=temp_arr[lvl]
                    )
                    total_loss += loss
                    levelwise_hit[lvl] += (y_pred.max(dim=-1)[1] == y).sum().item()

                sum_loss += total_loss.item()
                hit = levelwise_hit[levels]# - 1]
                cnt += args.v_batch_size

                if i == 0:
                    sample_imgs = x[:10]
                    length = len(t)
                    bbox_scaler = T.FloatTensor([[x.shape[3], x.shape[2], x.shape[3], x.shape[2]]]).to(x)
                    sample_bboxs = [
                            glimpse_to_xyhw(t[k].bbox[:10, :4].detach()) * bbox_scaler
                            for k in range(1, length)
                            ]
                    sample_g_arr = [t[_].g[:10] for _ in range(length)]
                    sample_atts = att_weights.cpu().numpy()[:10]

                    viz(epoch, sample_imgs, sample_bboxs, sample_g_arr, sample_atts, 'valid', n_branches=n_branches, n_levels=levels)

        avg_loss = sum_loss / i
        acc = hit * 1.0 / cnt
        levelwise_acc = levelwise_hit * 1.0 / cnt
        print("Loss on valid set: {}".format(avg_loss))
        print("Accuracy on valid set: {}".format(acc))

        writer.add_scalar('data/loss', avg_loss, epoch)
        if args.schedule:
            writer.add_scalars('data/levelwise_loss', {str(lvl): levelwise_acc[lvl] for lvl in range(levels + 1)}, epoch)
        writer.add_scalar('data/accuracy', acc, epoch)

        if (epoch + 1) % 10 == 0:
            print('Save checkpoint...')
            if not os.path.exists('checkpoints/'):
                os.makedirs('checkpoints')
            T.save(builder, 'checkpoints/{}_builder_{}.pt'.format(expr_setting, epoch))
            T.save(readout, 'checkpoints/{}_readout_{}.pt'.format(expr_setting, epoch))

        if best_valid_loss > avg_loss:
            best_valid_loss = avg_loss
            T.save(builder.state_dict(), 'checkpoints/{}_builder_best.pt'.format(expr_setting))
            T.save(readout.state_dict(), 'checkpoints/{}_readout_best.pt'.format(expr_setting))
            best_epoch = epoch
        elif best_epoch < epoch - 20 and lr > 1e-4 and args.dataset in dataset_with_sgd_schedule:
            best_epoch = epoch
            if levels < 2:
                print('Increasing level...')
                levels += 1
            else:
                print('Shrinking learning rate...')
                lr /= 10
                for pg in opt.param_groups:
                    pg['lr'] = lr
            builder.load_state_dict(T.load('checkpoints/{}_builder_best.pt'.format(expr_setting)))
            readout.load_state_dict(T.load('checkpoints/{}_readout_best.pt'.format(expr_setting)))
        elif (best_epoch < epoch - 10 or epoch == n_epochs - 1) and test_loader is not None:
            print('Early Stopping...')
            builder.load_state_dict(T.load('checkpoints/{}_builder_best.pt'.format(expr_setting)))
            readout.load_state_dict(T.load('checkpoints/{}_readout_best.pt'.format(expr_setting)))
            cnt = 0
            hit = 0
            levelwise_hit = np.zeros(n_levels + 1)
            sum_loss = 0
            with T.no_grad():
                for i, item in enumerate(tqdm.tqdm(test_loader)):
                    x, y, b = preprocessor(item)
                    if args.dataset in dataset_with_normalize:
                        x_in = imagenet_normalize(x)
                    else:
                        x_in = x

                    total_loss = 0
                    t, _ = builder(x_in, levels)
                    readout_list = readout(t, levels)

                    for lvl in range(readout_start_lvl, levels + 1):
                        y_pred, att_weights = readout_list[lvl]
                        loss = kl_temperature(
                            y_pred, y, temperature=temp_arr[lvl]
                        )
                        total_loss += loss
                        levelwise_hit[lvl] += (y_pred.max(dim=-1)[1] == y).sum().item()

                    sum_loss += total_loss.item()
                    hit = levelwise_hit[levels]
                    cnt += args.v_batch_size

            avg_loss = sum_loss / i
            acc = hit * 1.0 / cnt
            levelwise_acc = levelwise_hit * 1.0 / cnt
            print("Loss on test set: {}".format(avg_loss))
            print("Accuracy on test set: {}".format(acc))

            for lvl in range(levels + 1):
                print("Levelwise accuracy on level {}: {}".format(lvl, levelwise_acc[lvl]))
            break

if __name__ == '__main__':
    train()
