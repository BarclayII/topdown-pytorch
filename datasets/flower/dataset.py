import torch as T
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
import os
import numpy as np

class FlowerSingle(Dataset):
    def __init__(self,
                 mode='train'):
        self.path = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(self.path, '{}.pt'.format(mode)), 'rb') as f:
            dataset = T.load(f)
        self.data = dataset['data']
        self.labels = dataset['labels']

    def __len__(self):
        return self.data.shape[0]

    def __getitem__ (self, idx):
        return self.data[idx].permute(2, 0, 1) / 255, \
            self.labels[idx]

if __name__ == '__main__':
    flower = FlowerSingle(mode='train')
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    for i in range(1000):
        img, lbl = flower[i]
        if lbl.item() == 10:
            plt.imshow(img.permute(1, 2, 0))
            plt.show()
    """
    sampler = FlowerBatchSampler(flower)
    for batch in sampler:
        data, label = batch
        print(data)
        print(label)
        break
    """
