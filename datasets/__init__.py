import torchvision
from torchvision.transforms import ToTensor, RandomCrop, RandomHorizontalFlip, Normalize, Compose, ToPILImage, Resize, CenterCrop
from .mnist import MNISTMulti
from .wrapper import wrap_output
from .sampler import SubsetSampler
from .imagenet import ImageNetSingle, ImageNetBatchSampler
from .flower.dataset import FlowerSingle
from .bird.dataset import BirdSingle
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
    y = _y.sort(dim=-1)[0]
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
                            num_workers=num_workers,
                            )
    return dataloader

def preprocess_imagenet(item):
    _x, _y, _ = item
    return cuda(_x), cuda(_y.squeeze(1)), None

def data_generator_flower(dataset, batch_size, **config):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=config['shuffle'], drop_last=True, num_workers=0)
    return dataloader

def data_generator_bird(dataset, batch_size, **config):
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=config['sampler'], drop_last=True, num_workers=0)
    return dataloader

def preprocess_flower(item):
    _x, _y = item
    return cuda(_x), cuda(_y.squeeze(1)), None

def preprocess_bird(item):
    _x, _y = item
    return cuda(_x), cuda(_y.squeeze(1)), None

def get_generator(args):
    if args.dataset.startswith('mnist'):
        cluttered = args.dataset.endswith('cluttered')
        dataset_train = MNISTMulti('.', n_digits=args.n_digits, backrand=args.backrand, cluttered=cluttered, image_rows=args.row, image_cols=args.col, download=True, size_min=args.size_min, size_max=args.size_max)
        dataset_valid = MNISTMulti('.', n_digits=args.n_digits, backrand=args.backrand, cluttered=cluttered, image_rows=args.row, image_cols=args.col, download=False, mode='valid', size_min=args.size_min, size_max=args.size_max)
        dataset_test = MNISTMulti('.', n_digits=args.n_digits, backrand=args.backrand, cluttered=cluttered, image_rows=args.row, image_cols=args.col, download=False, mode='test', size_min=args.size_min, size_max=args.size_max)
        train_sampler = valid_sampler = test_sampler = None
        loader_train = data_generator_mnistmulti(dataset_train, args.batch_size, shuffle=True)
        loader_valid = data_generator_mnistmulti(dataset_valid, args.v_batch_size, shuffle=False)
        loader_test = data_generator_mnistmulti(dataset_test, args.v_batch_size, shuffle=False)
        preprocessor = preprocess_mnistmulti
    elif args.dataset == 'cifar10':
        # TODO: test set
        #transform_train = Compose([
        #    RandomCrop(32, padding=4),
        #    RandomHorizontalFlip(),
        #    ToTensor(),
        #])
        #transform_test = Compose([
        #    ToTensor(),
        #])
        dataset_train = torchvision.datasets.CIFAR10('.', download=True, transform=ToTensor())
        dataset_valid = torchvision.datasets.CIFAR10('.', download=True, transform=ToTensor())
        train_sampler = SubsetRandomSampler(range(0, 45000))
        valid_sampler = SubsetSampler(range(45000, 50000))
        loader_train = data_generator_cifar10(dataset_train, args.batch_size, sampler=train_sampler)
        loader_valid = data_generator_cifar10(dataset_valid, args.v_batch_size, sampler=valid_sampler)
        loader_test = None
        preprocessor = preprocess_cifar10
        args.row = args.col = 32
    elif args.dataset == 'bird':
        transform_train = Compose([
            ToPILImage(),
            RandomCrop(448),
            RandomHorizontalFlip(),
            ToTensor(),
        ])
        transform_test = Compose([
            ToPILImage(),
            CenterCrop(448),
            ToTensor(),
        ])
        dataset_train = BirdSingle('train', transform=transform_train)
        dataset_test = BirdSingle('test', transform=transform_test)
        train_sampler = SubsetRandomSampler(range(0, 3000))
        #valid_sampler = SubsetSampler(range(2700, 3000))
        test_sampler = SubsetSampler(range(0, 3033))
        loader_train = data_generator_bird(dataset_train, args.batch_size, sampler=train_sampler)
        #loader_valid = data_generator_bird(dataset_train, args.batch_size, sampler=valid_sampler)
        loader_test = data_generator_bird(dataset_test, args.v_batch_size, sampler=test_sampler)
        loader_valid = loader_test
        preprocessor = preprocess_bird
    elif args.dataset == 'flower':
        dataset_train = FlowerSingle('train')
        dataset_valid = FlowerSingle('valid')
        dataset_test = FlowerSingle('test')
        loader_train = data_generator_flower(dataset_train, args.batch_size, shuffle=True)
        loader_valid = data_generator_flower(dataset_valid, args.v_batch_size, shuffle=False)
        loader_test = data_generator_flower(dataset_test, args.v_batch_size, shuffle=False)
        preprocessor = preprocess_flower
    elif args.dataset in ['imagenet', 'dogs']:
        # TODO: test set
        dataset_train = ImageNetSingle(args.imagenet_root, args.imagenet_train_sel, args.batch_size)
        dataset_valid = ImageNetSingle(args.imagenet_root, args.imagenet_valid_sel, args.v_batch_size)
        train_sampler = ImageNetBatchSampler(dataset_train)
        valid_sampler = ImageNetBatchSampler(dataset_valid)
        loader_train = data_generator_imagenet(dataset_train, args.batch_size, num_workers=args.num_workers)
        loader_valid = data_generator_imagenet(dataset_valid, args.batch_size, num_workers=args.num_workers)
        loader_test = None
        preprocessor = preprocess_imagenet

    return loader_train, loader_valid, loader_test, preprocessor
