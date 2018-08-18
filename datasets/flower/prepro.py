import torch as T
import scipy.io
import pickle
import os
import cv2
import numpy as np

label_list = scipy.io.loadmat('imagelabels.mat')['labels'][0]
setid_dict = scipy.io.loadmat('setid.mat')
id_list = {}
id_list['train'] = setid_dict['trnid'][0]
id_list['valid'] = setid_dict['valid'][0]
id_list['test'] = setid_dict['tstid'][0]

id_dict = {}
for mode in ['train', 'valid', 'test']:
    id_dict[mode] = {
        v: k for k, v in enumerate(id_list[mode])
    }

row = 200 
col = 200

dataset = {}
labels = {}

for mode in ['train', 'valid', 'test']:
    length = len(id_list[mode])
    dataset[mode] = np.zeros((length, row, col, 3))
    labels[mode] = np.zeros((length, 1), dtype=np.int32)

img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'jpg')

from tqdm import tqdm
for filename in tqdm(os.listdir(img_dir)):
    im = cv2.resize(
        cv2.imread(os.path.join(img_dir, filename)),
        (col, row)
    )
    idx = int(filename[7:11])
    for mode in ['train', 'valid', 'test']:
        if idx in id_dict[mode]:
            pos = id_dict[mode][idx]
            dataset[mode][pos] = im
            labels[mode][pos] = label_list[idx - 1] - 1

for mode in ['train', 'valid', 'test']:
    T.save({
        'data': T.FloatTensor(dataset[mode]),
        'labels': T.LongTensor(labels[mode])
    }, '{}.pt'.format(mode))

print('done...')
