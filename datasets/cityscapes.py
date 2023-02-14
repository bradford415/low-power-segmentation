from torch.utils.data import Dataset
import os
import random
import glob
from PIL import Image
from utils import preprocess


class Cityscapes(Dataset):
  CLASSES = [
      'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
      'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
      'truck', 'bus', 'train', 'motorcycle', 'bicycle'
  ]

  def __init__(self, root, train=True, transform=None, target_transform=None, crop_size=None):
    self.root = root
    self.transform = transform
    self.target_transform = target_transform
    self.train = train
    self.crop_size = crop_size

    dataset_split = 'train' if self.train else 'val' # Cityscapes does not have a public test set
    self.images = self._get_files(dataset_split, 'leftImg8bit')
    self.masks = self._get_files(dataset_split, 'gtFine')

  def __getitem__(self, index):
    _img = Image.open(self.images[index]).convert('RGB')
    _target = Image.open(self.masks[index])

    _img, _target = preprocess(_img, _target,
                               flip=True if self.train else False,
                               scale=(0.5, 2.0) if self.train else None,
                               crop=(self.crop_size, self.crop_size) if self.train else (1025, 2049))

    if self.transform is not None:
      _img = self.transform(_img)

    if self.target_transform is not None:
      _target = self.target_transform(_target)

    return _img, _target

  def _get_files(self, dataset_split, data_type):
    dataset_path = os.path.join(self.root, data_type, dataset_split)
    
    file_ending = 'labelTrainIds.png' if data_type == 'gtFine' else 'leftImg8bit.png'
    filenames = glob.glob(f'{self.root}/{data_type}/{dataset_split}/*/*{file_ending}')
    return sorted(filenames)

  def __len__(self):
    return len(self.images)
    