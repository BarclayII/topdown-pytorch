import torchvision
from torchvision.transforms import ToTensor
from .mnist import MNISTMulti
from .wrapper import wrap_output
from .sampler import SubsetSampler
from .imagenet import ImageNetSingle, ImageNetBatchSampler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from util import cuda
from functools import partial

def data_generator_mnistmulti(dataset, batch_size, **config):
    shuffle = config['shuffle']
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=0)
    return dataloader

def preprocess_mnistmulti(item):
    _x, _y, _B = item
    x = _x[:, None].expand(_x.shape[0], 3, _x.shape[1], _x.shape[2]).float() / 255.
    y = _y
    n_digits = y.shape[1]
    new_y = y[:, 0]
    for digit in range(1, n_digits):
        new_y = new_y * 10 + y[:, digit]
    y = new_y
    b = _B.squeeze(1).float()
    return cuda(x), cuda(y), cuda(b)

def data_generator_cifar10(dataset, batch_size, **config):
    sampler = config['sampler']
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True, num_workers=0)
    return dataloader

def preprocess_cifar10(item):
    _x, _y = item
    return cuda(_x), cuda(_y), None

def data_generator_imagenet(dataset, batch_size, **config):
    num_workers = config['num_workers']
    dataloader = DataLoader(dataset,
                            batch_sampler=ImageNetBatchSampler(dataset),
                            num_workers=8,
                            )
    return dataloader

def preprocess_imagenet(item):
    _x, _y, _ = item
    return cuda(_x), cuda(_y.squeeze(1)), None

def get_generator(args):
    if args.dataset.startswith('mnist'):
        cluttered = args.dataset.endswith('cluttered')
        dataset_train = MNISTMulti('.', n_digits=args.n_digits, backrand=args.backrand, cluttered=cluttered, image_rows=args.row, image_cols=args.col, download=True, size_min=args.size_min, size_max=args.size_max)
        dataset_valid = MNISTMulti('.', n_digits=args.n_digits, backrand=args.backrand, cluttered=cluttered, image_rows=args.row, image_cols=args.col, download=False, mode='valid', size_min=args.size_min, size_max=args.size_max)
        train_sampler = valid_sampler = None
        loader_train = data_generator_mnistmulti(dataset_train, args.batch_size, shuffle=True)
        loader_valid = data_generator_mnistmulti(dataset_valid, args.v_batch_size, shuffle=False)
        preprocessor = preprocess_mnistmulti
    elif args.dataset == 'cifar10':
        dataset_train = dataset_valid = torchvision.datasets.CIFAR10('.', download=True, transform=ToTensor())
        train_sampler = SubsetRandomSampler(range(0, 45000))
        valid_sampler = SubsetSampler(range(45000, 50000))
        loader_train = data_generator_cifar10(dataset_train, args.batch_size, sampler=train_sampler)
        loader_valid = data_generator_cifar10(dataset_valid, args.v_batch_size, sampler=valid_sampler)
        preprocessor = preprocess_cifar10
        args.row = args.col = 32
    elif args.dataset == 'imagenet':
        dataset_train = ImageNetSingle(args.imagenet_root, args.imagenet_train_sel, args.batch_size)
        dataset_valid = ImageNetSingle(args.imagenet_root, args.imagenet_valid_sel, args.v_batch_size)
        train_sampler = ImageNetBatchSampler(dataset_train)
        valid_sampler = ImageNetBatchSampler(dataset_valid)
        loader_train = data_generator_imagenet(dataset_train, args.batch_size, num_workers=args.num_workers)
        loader_valid = data_generator_imagenet(dataset_valid, args.batch_size, num_workers=args.num_workers)
        preprocessor = preprocess_imagenet

    return loader_train, loader_valid, preprocessor
