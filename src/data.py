#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from tqdm import tqdm
import random
# import torchvision.datasets.ImageFolder as datasets
from torchvision.datasets import ImageFolder
import random
from logging import getLogger
import argparse
from PIL import ImageFilter
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import jsonlines
from PIL import Image

def write_tsv(tsv, path):
    with open(path, "w") as f:
        for line in tsv:
            f.write("\t".join(json.dumps(seg, ensure_ascii=False) for seg in line) + "\n")

def get_tsv(label_list, pred_list, image_list, new_out_features_list=None, old_out_features_list=None):
    res = []
    len_preds_list = len(pred_list)

    for i in range(len_preds_list):
        if new_out_features_list != None and old_out_features_list != None:

            res.append((image_list[i], pred_list[i], label_list[i]), new_out_features_list[i], old_out_features_list[i])
        else:
            res.append((image_list[i], pred_list[i], label_list[i]))


    # logging.debug(f"[{dataset_name}] predict result save at {outputdir}")
    return res


class data_loader:
    def __init__(self, jsonl_path, transforms):
        super(data_loader, self).__init__()
        self.samples = []
        self.transforms = transforms
        jsonl_path = os.path.join(jsonl_path)
        with open(jsonl_path, "r", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                if os.path.isfile(item['origin_image_path']):
                    self.samples.append((item['origin_image_path'], item['label']))
#        print(self.samples)
        self.image_name_list = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        self.loader = default_loader
        sample = self.loader(path)
        if self.transforms:
            sample = self.transforms(sample)

        self.image_name_list.append(path)

        return sample, target

    def clear_image_list(self):
        self.image_name_list = []



def pil_loader(path):
    try:
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')
    except:
        print(path)

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)




if __name__ == '__main__':
    pass
