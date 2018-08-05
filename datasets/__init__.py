import torchvision
from torchvision.transforms import ToTensor
from .mnist import MNISTMulti
from .wrapper import wrap_output
from .sampler import SubsetSampler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from util import cuda

def data_generator_mnistmulti(dataset, batch_size, **config):
    shuffle = config['shuffle']
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=0)
    for _x, _y, _B in dataloader:
        x = _x[:, None].expand(_x.shape[0], 3, _x.shape[1], _x.shape[2]).float() / 255.
        y = _y.squeeze(1)
        n_digits = y.shape[1]
        if n_digits == 2:
            y = y[:, 0] * 10 + y[:, 1]
        b = _B.squeeze(1).float() / 200
        yield cuda(x), cuda(y), cuda(b)

def data_generator_cifar10(dataset, batch_size, **config):
    sampler = config['sampler']
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True, num_workers=0)
    for _x, _y in dataloader:
        yield cuda(_x), cuda(_y), None

def get_generator(args):
    if args.dataset == 'mnistmulti':
        dataset_train = MNISTMulti('.', n_digits=args.n_digits, backrand=args.backrand, image_rows=args.row, image_cols=args.col, download=True, size_min=args.size_min, size_max=args.size_max)
        dataset_valid = MNISTMulti('.', n_digits=args.n_digits, backrand=args.backrand, image_rows=args.row, image_cols=args.col, download=False, mode='valid', size_min=args.size_min, size_max=args.size_max)
        train_sampler = valid_sampler = None
        data_generator = data_generator_mnistmulti
    elif args.dataset == 'cifar10':
        dataset_train = dataset_valid = torchvision.datasets.CIFAR10('.', download=True, transform=ToTensor())
        train_sampler = SubsetRandomSampler(range(0, 45000))
        valid_sampler = SubsetSampler(range(45000, 50000))
        data_generator = data_generator_cifar10

    return data_generator, dataset_train, dataset_valid, train_sampler, valid_sampler
