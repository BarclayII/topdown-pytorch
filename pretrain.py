import torch as T
import torch.nn.functional as F
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip
from datasets import get_generator
from modules import WhatModule, InverseGlimpse
from glimpse import GaussianGlimpse, BilinearGlimpse
import pytorch_cifar.models
import torchvision.models
from util import USE_CUDA, cuda, imagenet_normalize
import numpy as np
import argparse
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--cnn', default='custom', help='(custom, ...)')
parser.add_argument('--dataset', default='cifar10', help='(cifar10, mnist)')
parser.add_argument('--batch_size', default=8, help='(cifar10, mnist)', type=int)
parser.add_argument('--v_batch_size', default=8, help='(cifar10, mnist)', type=int)
parser.add_argument('--imagenet_train_sel', default='selected-train.pkl', type=str)
parser.add_argument('--imagenet_valid_sel', default='selected-val.pkl', type=str)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--imagenet_root', default='/beegfs/qg323', type=str)
args = parser.parse_args()

train_dataloader, valid_dataloader, test_dataloader, preprocessor = get_generator(args)

#mnist = MNISTMulti('.', n_digits=1, backrand=0, image_rows=200, image_cols=200, download=True)

#dfs = DFSGlimpseSingleObjectClassifier()
#dfs.load_state_dict(T.load('bigmodel.pt'))

n_glimpses = 1

if args.cnn == 'custom':
    cnn = WhatModule(
            [16, 32, 64, 128, 256],
            (3, 3),
            (2, 2),
            128,
            10,
            )
    module = T.nn.Sequential(
            MultiscaleGlimpse(glimpse_type='gaussian', glimpse_size=(15, 15), n_glimpses=n_glimpses),
            cnn,
            )
    #module = cnn
else:
    #cnn = miniresnet20(num_classes=10)
    #cnn = getattr(pytorch_cifar.models, args.cnn)(1000)
    cnn = getattr(torchvision.models, args.cnn)(pretrained=True)
    cnn.fc = T.nn.Linear(512 * 4, 120)
    module = T.nn.DataParallel(T.nn.Sequential(
            #MultiscaleGlimpse(glimpse_type='gaussian', glimpse_size=(50, 50), n_glimpses=n_glimpses),
            cnn,
            ))

module = cuda(module)
#module.load_state_dict(T.load('cnn.pt'))
#module.load_state_dict(dfs.update_module.cnn.state_dict())

'''
net = skorch.NeuralNetClassifier(
        module=module,
        #module=CNN,
        #module__cnn='cnn',
        #module__input_size=(15, 15),
        #module__h_dims=128,
        #module__n_classes=10,
        #module__kernel_size=(3, 3),
        #module__final_pool_size=(2, 2),
        #module__filters=[16, 32, 64, 128, 256],
        criterion=T.nn.CrossEntropyLoss,
        max_epochs=50,
        optimizer=T.optim.Adam,
        #optimizer__param_groups=[
        #    ('cnn.*', {'lr': 0}),
        #    ('net_h.*', {'lr': 0}),
        #    ],
        optimizer__weight_decay=1e-4,
        batch_size=32,
        device='cuda' if USE_CUDA else 'cpu',
        callbacks=[
            skorch.callbacks.EpochScoring('accuracy', name='train_acc', on_train=True, lower_is_better=False),
            skorch.callbacks.ProgressBar(),
            skorch.callbacks.Checkpoint('cnntest.pt'),
            ],
        )
'''
#net.fit(train_data, train_labels, epochs=100)

#opt = T.optim.Adam(module.parameters(), weight_decay=1e-4)

g = cuda(T.nn.DataParallel(GaussianGlimpse((50, 50))))

lr = 0.1
#opt = T.optim.Adam(module.parameters(), weight_decay=5e-4)

best_acc = 0
best_valid_loss = 1e6
best_epoch = 0
n_iter = 0
for epoch in range(2000):
    if epoch == 0:
        opt = T.optim.SGD(module.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    #elif epoch == 1:
    #    opt = T.optim.SGD(module.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    correct = 0
    total = 0
    avg_loss = 0
    batches = 0
    with tqdm.tqdm(train_dataloader) as t:
        for item in t:
            x, y, b = preprocessor(item)
            x = imagenet_normalize(x)
            #g1 = cuda(T.nn.DataParallel(BilinearGlimpse((x.shape[2], x.shape[3]))))
            #b = g.module.full()[None, None, :].expand(args.batch_size, 1, 6)
            #b1 = g1.module.full()[None, None, :].expand(args.batch_size, 1, 4)
            #x = g1(g(x, b)[:, 0], b1)[:, 0]
            n_iter += 1
            opt.zero_grad()
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            pred = module(x)
            loss = F.cross_entropy(pred, y)
            loss.backward()
            correct += (pred.max(1)[1] == y).sum().item()
            total += x.shape[0]
            avg_loss = (avg_loss * batches + loss.item()) / (batches + 1)
            batches += 1
            opt.step()

            t.set_postfix({'train_acc': correct / total, 'train_loss': avg_loss})
    correct = 0
    total = 0
    avg_loss = 0
    batches = 0
    with tqdm.tqdm(valid_dataloader) as t:
        with T.no_grad():
            for item in t:
                x, y, b = preprocessor(item)
                x = imagenet_normalize(x)
                #g1 = cuda(T.nn.DataParallel(BilinearGlimpse((x.shape[2], x.shape[3]))))
                #b = g.module.full()[None, None, :].expand(args.batch_size, 1, 6)
                #b1 = g1.module.full()[None, None, :].expand(args.batch_size, 1, 4)
                #x = g1(g(x, b)[:, 0], b1)[:, 0]
                if x.shape[1] == 1:
                    x = x.repeat(1, 3, 1, 1)
                pred = module(x)
                loss = F.cross_entropy(pred, y)
                correct += (pred.max(1)[1] == y).sum().item()
                total += x.shape[0]
                avg_loss = (avg_loss * batches + loss.item()) / (batches + 1)
                batches += 1

                t.set_postfix({'valid_acc': correct / total, 'valid_loss': avg_loss})

    if best_acc < correct / total:
        best_acc = correct / total
        T.save(cnn.state_dict(), 'cnntest.pt')
    if best_valid_loss > avg_loss:
        best_valid_loss = avg_loss
        best_epoch = epoch
    elif best_epoch < epoch - 20:
        if lr > 0.0001:
            best_epoch = epoch
            print('Shrinking learning rate...')
            lr /= 10
            opt = T.optim.SGD(module.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
