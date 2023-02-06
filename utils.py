"""Helper functions for the model
 
"""
import math
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


class AverageMeter(object):
    def __init__(self):
        self.val = None
        self.sum = None
        self.count = None
        self.avg = None
        self.ema = None  # ema = exponential moving averages
        self.initialized = False

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def initialize(self, val, n):
        self.val = val
        self.sum = val * n 
        self.count = n
        self.avg = val
        self.ema = val
        self.initialized = True

    def add(self, val, n):
        self.val = val
        self.sum += val * n # count of pixels per class (used for intersection and union)
        self.count += n # n = number of images being passed into AverageMeter (usually always 1)
        self.avg = self.sum / self.count
        self.ema = self.ema * 0.99 + self.val * 0.01


def inter_and_union(pred, mask, num_class):
    pred = np.asarray(pred, dtype=np.uint8).copy()
    mask = np.asarray(mask, dtype=np.uint8).copy()

    # 255 -> 0
    pred += 1
    mask += 1
    pred = pred * (mask > 0)

    inter = pred * (pred == mask)
    (area_inter, _) = np.histogram(inter, bins=num_class, range=(1, num_class))
    (area_pred, _) = np.histogram(pred, bins=num_class, range=(1, num_class))
    (area_mask, _) = np.histogram(mask, bins=num_class, range=(1, num_class))
    area_union = area_pred + area_mask - area_inter

    return (area_inter, area_union)


def preprocess(image, mask, flip=False, scale=None, crop=None):
    """Preprocess images as defined in the deeplabv3 paper. This includes
    random resizing from 0.5-2.0, horizontal flipping, random cropping, and 
    normalizing the values based on the mean and standard deviation of the 
    pretrained network dataset (ImageNet)

    MAKE SURE YOU PERFORM THE SAME TRANSFORM, WITH THE SAME TRANSFORM VALUES, 
    FOR THE IMAGE AND LABEL. It is shown here how to do this:
    https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/7
    
    The preprocessing protocol is defined by the original protocol of deeplab.
    The idea behind horizontal flipping is that an object should be equally
    identifiable if it were reversed, however that is not the same for vertical
    flipping. This same idea applies scaling and random cropping.

    Finally, normalizing an image based on the mean and standard deviation of 
    the original dataset is common practice. If you are using pretrained weights
    for a network, you want to normalize your dataset using the values from the
    original dataset. In most cases it will be the ImageNet dataset as that is
    what most networks are pretrained on. First, normalize your dataset from 0-1 
    (ToTensor()), and then again normalize it using the mean and std dev for each 
    channel (Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    The mean and standard deviation are given from the ImageNet dataset.

    Training applies crop, flip, resize, and nromalize
    """
    if flip:
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    if scale:
        w, h = image.size
        rand_log_scale = math.log(scale[0], 2) + random.random() * (
            math.log(scale[1], 2) - math.log(scale[0], 2))
        random_scale = math.pow(2, rand_log_scale)
        new_size = (int(round(w * random_scale)), int(round(h * random_scale)))
        image = image.resize(new_size, Image.ANTIALIAS)
        mask = mask.resize(new_size, Image.NEAREST)

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = data_transforms(image)
    mask = torch.LongTensor(np.array(mask).astype(np.int64))

    if crop:
        h, w = image.shape[1], image.shape[2]
        pad_tb = max(0, crop[0] - h)
        pad_lr = max(0, crop[1] - w)
        image = torch.nn.ZeroPad2d((0, pad_lr, 0, pad_tb))(image)
        mask = torch.nn.ConstantPad2d((0, pad_lr, 0, pad_tb), 255)(mask)

        h, w = image.shape[1], image.shape[2]
        i = random.randint(0, h - crop[0])
        j = random.randint(0, w - crop[1])
        image = image[:, i:i + crop[0], j:j + crop[1]]
        mask = mask[i:i + crop[0], j:j + crop[1]]

    return image, mask
