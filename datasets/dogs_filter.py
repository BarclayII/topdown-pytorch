#!/usr/bin/python3
# Usage: ./prepro.py [path-to-imagenet-ILSVRC-directory]
# It will generate two pickle files containing two pandas DataFrames
from lxml import etree
import os
import sys
import pickle
from PIL import Image
import pandas as pd
import scipy.io as sio

parser = etree.XMLParser(remove_blank_text=True, remove_comments=True)

ilsvrc_root = sys.argv[1]
ilsvrc_anno_root = os.path.join(ilsvrc_root, 'Annotation')
ilsvrc_data_root = os.path.join(ilsvrc_root, 'Images')

for dataset in ['train', 'test']:
    total_anno_files = 0
    selected_anno_files = []
    mat = sio.loadmat(os.path.join(ilsvrc_root, dataset + '_list.mat'))

    for img_entry, anno_entry in zip(mat['file_list'], mat['annotation_list']):
        path = os.path.join(ilsvrc_anno_root, anno_entry[0][0])
        anno = etree.parse(path, parser=parser).getroot()
        size = anno.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        img_path = os.path.join(ilsvrc_data_root, img_entry[0][0])

        img = Image.open(img_path)
        assert img.width == width
        assert img.height == height

        objs = anno.findall('object')
        if len(objs) == 1:
            for obj in objs:
                name = obj.find('name').text
                xmin, xmax, ymin, ymax = [
                        int(obj.find('bndbox').find(tag).text)
                        for tag in ['xmin', 'xmax', 'ymin', 'ymax']
                        ]
                obj_width = xmax - xmin
                obj_height = ymax - ymin
                ratio = (obj_width * obj_height) / (width * height)
                selected_anno_files.append({
                        'annopath': os.path.join('Annotation', anno_entry[0][0]),
                        'imgpath': os.path.join('Images', img_entry[0][0]),
                        'name': name,
                        'ratio': ratio,
                        'imgwidth': width,
                        'imgheight': height,
                        'xmin': xmin,
                        'xmax': xmax,
                        'ymin': ymin,
                        'ymax': ymax,
                        })
        total_anno_files += 1
        print('\x1b[2K', len(selected_anno_files), '/', total_anno_files, end='\r')

    print()
    selected_anno_files = pd.DataFrame(selected_anno_files)
    with open('selected-%s.pkl' % dataset, 'wb') as f:
        pickle.dump(selected_anno_files, f)
