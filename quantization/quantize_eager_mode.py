"""Post Training Static Quantization in PyTorch using Eager Mode
https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
"""
import os
import sys
import time
import numpy as np
import copy

sys.path.append('../') # make parent-level modules visible to this dir 

import torch
from torch.ao.quantization import get_default_qconfig, QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, fuse_fx
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
import torchvision.transforms as transforms

from networks import create_quant_resnet101
from datasets import lpcvc
from utils.utils import AverageMeter, inter_and_union

# Set up warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.ao.quantization'
)


# Specify random seed for repeatable results
#_ = torch.manual_seed(191009)

def load_model(model_file, device='cpu'):

    model = create_quant_resnet101(
        pretrained=False,
        device=device,
        num_classes=14)

    weights = torch.load(model_file, map_location=device)

    # Remove 'module.' appended by DataParallel() 
    state_dict = {k[7:]: v for k,
                v in weights['model'].items()}
    # Do not need to load optimizer state_dict because it is not used for inference
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    print(f"Loading model {model_file}")
    return model


def print_size_of_model(model):
    torch.save(model, "temp.p")
    print("Size (MB):", os.path.getsize("temp.p")/1e6)
    os.remove("temp.p")


def prepare_data_loaders(data_path):

    train_kwargs = {'batch_size': 8, 
                    'shuffle': True}
    val_kwargs = {'batch_size': 1,
                  'shuffle': False}  # val and test


    dataset_train = lpcvc(data_path,
                           train=True, crop_size=512)
    dataset_val = lpcvc(data_path,
                           train=False)

    loader_train = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
    loader_val = torch.utils.data.DataLoader(dataset_val, **val_kwargs)


    return loader_train, loader_val


def evaluate(model, loader, device='cpu', num_classes=14, neval_batches=None):
    inter_meter = AverageMeter()  
    union_meter = AverageMeter() 
    with torch.inference_mode():  # newer torch.no_grad()
        for index, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, pred = torch.max(outputs, 1)
            
            pred, target= pred.cpu(), target.cpu()
            print('eval: {0}/{1}'.format(index + 1, len(loader)))
            inter, union = inter_and_union(
                pred, target, num_classes)
            # Keep running sum of intersection and union values of image
            # Inter and union are based on the prediction and groud truth mask
            inter_meter.update(inter)
            union_meter.update(union)
            if  neval_batches is not None and index >= neval_batches:
                break
        iou = inter_meter.sum / (union_meter.sum + 1e-10) # 1e-10 is used to prevent division by 0 I think
        # Print and save IoU per class and final mIoU score
        print('Mean IoU: {0:.2f}'.format(iou.mean() * 100))



data_root = '/home/eceftl7/datasets/lpcvc'
model_path = '../trained-models/model_best_miou_54-85.pt'


train_loader, val_loader = prepare_data_loaders(data_root)
example_inputs = (next(iter(train_loader))[0])
criterion = nn.CrossEntropyLoss()
float_model = load_model(model_path, 'cpu')
float_model.eval()

print("Fusing model...")
float_model.fuse_model()

print_size_of_model(float_model)

#evaluate(float_model, val_loader, 'cuda')

# Quantize
num_calibration_batches = 128
model_to_quantize = copy.deepcopy(float_model)

#model_to_quantize.qconfig = torch.ao.quantization.default_qconfig # min-max quant
model_to_quantize.qconfig = torch.ao.quantization.get_default_qconfig('x86')
print(model_to_quantize.qconfig)
torch.ao.quantization.prepare(model_to_quantize, inplace=True)

# Calibrate first
print('Post Training Quantization Prepare: Inserting Observers')
# Calibrate with the training set
evaluate(model_to_quantize, train_loader, 'cpu', neval_batches=num_calibration_batches)
print('Post Training Quantization: Calibration done')

# Convert to quantized model
torch.ao.quantization.convert(model_to_quantize, inplace=True)
print('Post Training Quantization: Convert done')

print("Size of model after quantization")
print_size_of_model(model_to_quantize)

evaluate(model_to_quantize, val_loader, 'cpu')

torch.save({'model': model_to_quantize.state_dict()}, 'model_quantized.pt')





