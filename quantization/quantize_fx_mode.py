"""Post Training Static Quantization in PyTorch using FX Graph Mode
https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html

Currently unable to trace certain layers. Will try eager mode.
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

from networks import create_resnet101
from datasets import lpcvc


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
_ = torch.manual_seed(191009)

def load_model(model_file):

    device = 'cuda'
        
    model = create_resnet101(
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
    if isinstance(model, torch.jit.RecursiveScriptModule):
        torch.jit.save(model, "temp.p")
    else:
        torch.jit.save(torch.jit.script(model), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p")/1e6)
    os.remove("temp.p")


def prepare_data_loaders(data_path):

    train_kwargs = {'batch_size': 4, 
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

data_root = '/home/eceftl7/datasets/lpcvc'
model_path = '../trained-models/model_best_miou_54-85.pt'

train_loader, val_loader = prepare_data_loaders(data_root)
example_inputs = (next(iter(train_loader))[0])

float_model = load_model(model_path)
float_model.eval()

model_to_quantize = copy.deepcopy(float_model)

qconfig_mapping = QConfigMapping().set_global(torch.ao.quantization.default_qconfig)

prepared_model = prepare_fx(model_to_quantize, qconfig_mapping, example_inputs)



