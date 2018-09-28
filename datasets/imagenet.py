import pandas as pd
import pickle
import torch as T
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision.transforms import RandomResizedCrop, ColorJitter
from util import *
from PIL import Image
import os
import numpy as np

class ImageNetSingle(Dataset):
    def __init__(self,
                 rootdir='.',
                 selection='selected-train.pkl',
                 batch_size=32,
                 batches_per_page=4,
                 bbox_shrink_ratio=0.05,
                 allowed_size=400):
        self.rootdir = rootdir
        with open(selection, 'rb') as f:
            self.selection = pickle.load(f)
        self.selection = self.selection.astype({'name': 'category'})
        self.selection['cat_id'] = self.selection['name'].cat.codes
        self.selection['imgaspect'] = self.selection['imgwidth'] / self.selection['imgheight']
        self.batch_size = batch_size
        self.batches_per_page = batches_per_page
        page_size = batch_size * batches_per_page
        self.n_pages = len(self.selection) // page_size
        self.pages = np.array_split(np.argsort(self.selection['imgaspect']), self.n_pages)
        self.target_aspect = {}
        for p in self.pages:
            a = np.median(self.selection['imgaspect'][p])
            self.target_aspect.update((k, a) for k in p)

        self.page_size = page_size
        self.bbox_shrink_ratio = bbox_shrink_ratio
        self.allowed_size = allowed_size

    def __len__(self):
        return len(self.selection)

    def __getitem__(self, i):
        record = self.selection.iloc[i]
        imgpath = os.path.join(self.rootdir, record['imgpath'])
        _img = Image.open(imgpath)
        if self.target_aspect[i] < 1:
            height = self.allowed_size
            width = int(height * self.target_aspect[i])
        else:
            width = self.allowed_size
            height = int(width / self.target_aspect[i])

        _img = _img.convert('RGB').resize((width, height), Image.BILINEAR)
        
        _resizer = RandomResizedCrop(min(width, height))
        _jitterer = ColorJitter(0.1, 0.1, 0.1, 0.05)
        _img = _jitterer(_resizer(_img))

        img = T.FloatTensor(np.array(_img)).permute(2, 0, 1) / 255.
        _img.close()

        return img, \
               T.LongTensor([record.cat_id]), \
               T.FloatTensor([[(record.xmin + record.xmax) / 2,
                               (record.ymin + record.ymax) / 2,
                               (record.xmax - record.xmin),
                               (record.ymax - record.ymin),]])


class ImageNetBatchSampler(BatchSampler):
    def __init__(self, imagenet_dataset):
        self.batch_size = imagenet_dataset.batch_size
        self.batches_per_page = imagenet_dataset.batches_per_page
        self.pages = imagenet_dataset.pages

    def __iter__(self):
        for p in self.pages:
            q = np.random.permutation(p)
            for i in range(self.batches_per_page):
                yield q[i*self.batch_size:(i+1)*self.batch_size].tolist()

    def __len__(self):
        return len(self.pages) * self.batches_per_page
