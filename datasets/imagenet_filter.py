#!/usr/bin/python3
# Usage: ./prepro.py [path-to-imagenet-ILSVRC-directory]
# It will generate two pickle files containing two pandas DataFrames
from lxml import etree
import os
import sys
import pickle
from PIL import Image
import pandas as pd

parser = etree.XMLParser(remove_blank_text=True, remove_comments=True)

ilsvrc_root = sys.argv[1]
ilsvrc_anno_root = os.path.join(ilsvrc_root, 'Annotations', 'CLS-LOC')
ilsvrc_data_root = os.path.join(ilsvrc_root, 'Data', 'CLS-LOC')

for dataset in ['train', 'val']:
    total_anno_files = 0
    selected_anno_files = []

    ilsvrc_anno_dir = os.path.join(ilsvrc_anno_root, dataset)
    if dataset == 'train':
        ilsvrc_data_dir = os.path.join(ilsvrc_data_root, dataset)
    else:
        ilsvrc_data_dir = ilsvrc_data_root

    for rootdir, subdirs, files in os.walk(ilsvrc_anno_dir):
        for filename in files:
            if filename.endswith('.xml'):
                path = os.path.join(rootdir, filename)
                anno = etree.parse(path, parser=parser).getroot()
                size = anno.find('size')
                width = int(size.find('width').text)
                height = int(size.find('height').text)

                img_folder = anno.find('folder').text
                if img_folder[0] != 'n' and dataset == 'train':
                    img_folder = 'n' + img_folder
                img_filename = anno.find('filename').text + '.JPEG'
                img_path = os.path.join(ilsvrc_data_dir, img_folder, img_filename)

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
                                'annopath': path,
                                'imgpath': img_path,
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
