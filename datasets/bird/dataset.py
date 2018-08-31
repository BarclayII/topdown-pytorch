import torch as T
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
import os
import numpy as np

class BirdSingle(Dataset):
    def __init__(self,
                 mode='train',
                 transform=lambda x: x
    ):
        self.path = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(self.path, '{}.pt'.format(mode)), 'rb') as f:
            dataset = T.load(f)
        self.data = dataset['data']
        self.labels = dataset['labels']
        self.trsf = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__ (self, idx):
        return self.trsf(self.data[idx].permute(2, 0, 1) / 255), \
            self.labels[idx]

if __name__ == '__main__':
    from torchvision.transforms import ToTensor, RandomCrop, RandomHorizontalFlip, Normalize, Compose, ToPILImage
    transform_train = Compose([
        ToPILImage(),
        RandomCrop(448),
        RandomHorizontalFlip(),
        ToTensor(),
    ])

    bird = BirdSingle(mode='test')#, transform=transform_train)
    print(len(bird))
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    for i in range(1000):
        img, lbl = bird[i]
        print(img)
        if lbl.item() == 0:
            print(img.shape)
            plt.imshow(img.permute(1, 2, 0))
            plt.show()
