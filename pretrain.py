import torch as T
import skorch
from torchvision.models import resnet18
from torchvision.datasets import MNIST, CIFAR10
from datasets import MNISTMulti
from modules import WhatModule, MultiscaleGlimpse
from util import USE_CUDA, cuda
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cnn', default='custom', help='(custom, resnet18)')
parser.add_argument('--dataset', default='cifar10', help='(cifar10, mnist)')
args = parser.parse_args()

if args.dataset == 'mnist':
    dataset = MNIST('.', download=True)
elif args.dataset == 'cifar10':
    dataset = CIFAR10('.', download=True)
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
elif args.cnn == 'resnet18':
    # FIXME: input size (32x32) is too small for resnet18
    cnn = resnet18()
    module = cnn

module = cuda(module)
#module.load_state_dict(T.load('cnn.pt'))
#module.load_state_dict(dfs.update_module.cnn.state_dict())

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
        optimizer=T.optim.RMSprop,
        #optimizer__param_groups=[
        #    ('cnn.*', {'lr': 0}),
        #    ('net_h.*', {'lr': 0}),
        #    ],
        lr=3e-5,
        batch_size=32,
        device='cuda' if USE_CUDA else 'cpu',
        callbacks=[
            skorch.callbacks.EpochScoring('accuracy', name='train_acc', on_train=True, lower_is_better=False),
            skorch.callbacks.ProgressBar(),
            skorch.callbacks.Checkpoint('cnntest.pt'),
            ]
        )
if args.dataset == 'mnist':
    train_data = dataset.train_data.float()[:, None].repeat(1, 3, 1, 1) / 255.
    train_labels = dataset.train_labels[:, 0]
elif args.dataset == 'cifar10':
    train_data = dataset.train_data.transpose(0, 3, 1, 2).astype('float32') / 255.
    train_labels = np.array(dataset.train_labels).astype('int64')
net.fit(train_data, train_labels, epochs=100)
