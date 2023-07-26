"""Dataset class for ADE20k semantic segmentation dataset
Dataset download: http://sceneparsing.csail.mit.edu/
Class labels and color maps: https://docs.google.com/spreadsheets/d/1se8YEtb2detS7OuPE86fXGyD269pMycAWe2mtKUj2W8/edit#gid=0
"""
from torch.utils.data import Dataset
import os
import random
import glob
from PIL import Image
from utils import preprocess
import numpy as np


class ade(Dataset):

  def __init__(self, root, num_classes, train=True, transform=None, target_transform=None, crop_size=None):
    self.root = root
    self.transform = transform
    self.target_transform = target_transform
    self.train = train
    self.crop_size = crop_size

    dataset_split = 'training' if self.train else 'validation'
    self.images = self._get_files(dataset_split, 'images')
    self.masks = self._get_files(dataset_split, 'annotations')
    self.num_samples = len(self.images)
    self.num_classes = num_classes

  def __getitem__(self, index):
    _img = Image.open(self.images[index]).convert('RGB')
    _target = Image.open(self.masks[index])

    _img, _target = preprocess(_img, _target,
                               flip=True if self.train else False,
                               scale=(0.5, 2.0) if self.train else None,
                               crop=(self.crop_size, self.crop_size) if self.train else self.crop_size) # if eval, self.crop_size=None (original image size)

    if self.transform is not None:
      _img = self.transform(_img)

    if self.target_transform is not None:
      _target = self.target_transform(_target)

    # Background class and 'other objects' are labeled 0 for ade20k (different from Cityscapes which uses 255)
    # A value of 255 still occurs after preprocessing due to the padding (torch.nn.ConstantPad2d)
    # Therefore, when we set the 0 pixels to 255 by subtracting 1, the padded pixels will become 254 and we must set them
    # back to 255
    # We must also ensure theses are unsigned 8 bit values so they range from 0-255 (0 - 1 = 255)

    _target[_target == 0] = 255
    _target = _target - 1
    _target[_target == 254] = 255
    #print("here4444")
    #print(np.unique(_target))
    #print(_target.shape)
    #print(np.savetxt('after.txt',_target))
    #print("done")
    #exit()

    return _img, _target

  def _get_files(self, dataset_split, data_type):
    dataset_path = os.path.join(self.root, data_type, dataset_split)
    
    file_end = '.png' if data_type == 'annotations' else '.jpg'
    filenames = glob.glob(f'{self.root}/{data_type}/{dataset_split}/*{file_end}')
    return sorted(filenames)

  def __len__(self):
    return len(self.images)
    