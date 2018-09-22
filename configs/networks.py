from .args import args
import torch.nn as nn

def _cifar10_cnn():
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
    return cnn


NETWORK_PARAMS = {
    'mnistcluttered': {
        'fm_target_size': (15, 15),
        'final_pool_size': (3, 3),
        'final_n_channels': 256,
        'n_classes': 10 ** args.n_digits,
        'cnn': None,
        'in_dims': None,
        'normalize': 'identity',
        'normalize_inverse': 'identity',
    },
    'mnistmulti': {
        'fm_target_size': (15, 15),
        'final_pool_size': (3, 3),
        'final_n_channels': 256,
        'n_classes': 10 ** args.n_digits,
        'cnn': None,
        'in_dims': None,
        'normalize': 'identity',
        'normalize_inverse': 'identity',
    },
    'bird': {
        'fm_target_size': (15, 15),
        'final_pool_size': (3, 3),
        'final_n_channels': 2048,
        'n_classes': 200,
        'cnn': 'resnet50',
        'in_dims': None,
        'normalize': 'imagenet_normalize',
        'normalize_inverse': 'imagenet_normalize_inverse',
    },
    'imagenet': {
        'fm_target_size': (15, 15),
        'final_pool_size': (3, 3),
        'final_n_channels': 2048,
        'n_classes': 1000,
        'cnn': 'resnet18',
        'in_dims': None,
        'normalize': 'imagenet_normalize',
        'normalize_inverse': 'imagenet_normalize_inverse',
    },
    'dogs': {
        'fm_target_size': (15, 15),
        'final_pool_size': (3, 3),
        'final_n_channels': 2048,
        'n_classes': 120,
        'cnn': 'resnet50',
        'in_dims': None,
        'normalize': 'imagenet_normalize',
        'normalize_inverse': 'imagenet_normalize_inverse',
    },
    'cifar10': {
        'fm_target_size': (15, 15),
        'final_pool_size': (3, 3),
        'final_n_channels': 2048,
        'n_classes': 10,
        'cnn': _cifar10_cnn,
        'in_dims': None,
        'normalize': 'imagenet_normalize',
        'normalize_inverse': 'imagenet_normalize_inverse',
    },
    'flower': {
        'fm_target_size': (15, 15),
        'final_pool_size': (3, 3),
        'final_n_channels': 2048,
        'n_classes': 102,
        'cnn': 'resnet18',
        'in_dims': None,
        'normalize': 'imagenet_normalize',
        'normalize_inverse': 'imagenet_normalize_inverse',
    },
}
