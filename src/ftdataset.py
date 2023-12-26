import enum
import random
from logging import getLogger, raiseExceptions
import argparse
from PIL import ImageFilter
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import jsonlines
import os
from PIL import Image
import torch.utils.data as data
from sklearn import preprocessing
import cv2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class FtData(data.Dataset):
    def __init__(self, jsonl_path, label_list,transforms, size_dataset=-1,):

        self.transforms = transforms
        self.samples = []
        self.classlist = []
        self.diseases_dict = {}


        with open(label_list,'r') as f:
            diseases = f.readlines()
        
        for i, disease in enumerate(diseases):
            self.diseases_dict[disease.rstrip()] = i

        with open(jsonl_path, 'r') as f:
            for item in jsonlines.Reader(f):
                image_path = item['image_path']

                image_file = (image_path, str(item['label']))
 
                self.samples.append(image_file)

        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        img_path, label= self.samples[index]

        image = default_loader(img_path)
        label_idx = self.diseases_dict[label] if label in self.diseases_dict else -1

        if self.transforms:
            image = self.transforms(image)

    
        return image, label_idx

        


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')



def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


    

class feature_dataset(datasets.ImageFolder):
    def __init__(self, jsonl_path, label_list,transforms_origin,transforms_crop, size_dataset=-1,):

        self.transforms_origin = transforms_origin
        self.transforms_crop = transforms_crop
        self.samples = []
        self.classlist = []
        self.diseases_dict = {}


        with open(label_list,'r') as f:
            diseases = f.readlines()
        
        for i, disease in enumerate(diseases):
            self.diseases_dict[disease.rstrip()] = i

        with open(jsonl_path, 'r') as f:
            for item in jsonlines.Reader(f):
                image_file = (item['origin_image_path'], item['crop_image_path'],item['label'],item['to_crop_bbox'] if 'to_crop_bbox' in item else None)
                # image_file = (item['input_image_path'], item['info']['output_path'],item['info']['label'],item['info']['to_crop_bbox'] if 'to_crop_bbox' in item['info'] else None)
                self.samples.append(image_file)

        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        origin_img_path,crop_img_path, label,_ = self.samples[index]

        origin_image = default_loader(origin_img_path)
        crop_image = default_loader(crop_img_path)
        label_idx = self.diseases_dict[label] if label in self.diseases_dict else -1
        if self.transforms_origin:
            origin_image = self.transforms_origin(origin_image)
        if self.transforms_crop:
            crop_image = self.transforms_crop(crop_image)
        return origin_image,crop_image, label_idx