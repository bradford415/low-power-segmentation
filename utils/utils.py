"""Helper functions for the model
"""
import math
import random
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image


class AverageMeter(object):
    """Stores loss and intersection/union pixel values"""
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
        """
        params:
            val: loss value of each mini-batch
            n: batch size
        """
        self.val = val # current value of update
        self.sum += val * n # estimates loss per sample (n=batch_size)
        self.count += n 
        self.avg = self.sum / self.count # average loss per epoch
        self.ema = self.ema * 0.99 + self.val * 0.01


def inter_and_union(pred, mask, num_class):
    """Returns intersection and union of pixels"""
    # Explanation of this at: https://github.com/bradford415/deeplabv3-pytorch/blob/main/utils.py
    pred = np.asarray(pred, dtype=np.uint8).copy()
    mask = np.asarray(mask, dtype=np.uint8).copy()

    # 255 -> 0
    pred += 1
    mask += 1
    pred = pred * (mask > 0)

    inter = pred * (pred == mask)
    (area_inter, _) = np.histogram(inter, bins=num_class, range=(1, num_class)) # TP
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
    
    The mean and standard deviation from the ImageNet dataset are used because we pretrain
    deeplab on ImageNet.

    Training applies random crop, flip, resize, and normalization
    """
    if flip:
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    # Resize image dimensions by a random factor between 0.5-2.0
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
        pad_tb = max(0, crop[0] - h) # Try to only crop within the image and not the padding
        pad_lr = max(0, crop[1] - w) # May crop the padding if the image size is smaller than the crop size
        image = torch.nn.ZeroPad2d((0, pad_lr, 0, pad_tb))(image)
        mask = torch.nn.ConstantPad2d((0, pad_lr, 0, pad_tb), 255)(mask) # 255 padding so we can ignore this index in the loss function

        h, w = image.shape[1], image.shape[2]
        i = random.randint(0, h - crop[0])
        j = random.randint(0, w - crop[1])
        image = image[:, i:i + crop[0], j:j + crop[1]]
        mask = mask[i:i + crop[0], j:j + crop[1]]

    #print(mask.shape)
        
    return image, mask


def colorize(image_path, prediction, save_name, cmap, labels):
    """Apply a colormap to the prediction"""
    prediction = prediction.numpy().squeeze().astype(np.uint8)
    pred_pil = Image.fromarray(prediction)
    pred_pil.putpalette(np.array(cmap).flatten().tolist())

    image = Image.open(image_path)
    #https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
    images = [image, pred_pil]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    # Horizontally concatenate raw and predicted image
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    # Label the predicted segmentation map
    colors = [ [color[0] / 255, color[1] / 255, color[2] / 255] 
               for color in cmap]
    labels = [label for label in labels]
    plt.figure(figsize = (9,9))
    ax = plt.imshow(new_im)
    plt.title(Path(image_path).stem)
    plt.axis('off')
    patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=labels[i]) ) for i in range(len(labels)) ]
    plt.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0, ncol=2, 
               handleheight=1.1, handlelength=2.3, labelspacing=1, columnspacing=0.8, fontsize='large')

    plt.savefig(save_name, bbox_inches='tight')    
    plt.close() # Save memory

