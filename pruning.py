import os
import sys
import copy
import torch
import torch.nn.utils.prune as prune
import warnings
from torchvision.models import resnet18
from torchvision.models import resnet101
import torch_pruning as tp
from torchsummary import summary

import argparse

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pdb  # Python debugger
import numpy as np
from scipy.io import loadmat
from PIL import Image
from pathlib import Path

# Import local files
from networks import deeplabv3
from utils import AverageMeter, inter_and_union, colorize
from datasets import VOCSegmentation
from datasets import Cityscapes
from datasets import Rellis3D


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, required=True,
                    help='Path to configuration file (.yaml file)')
args = parser.parse_args()

with open(args.cfg, "r") as stream:
    try:
        cfg = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

def depGraph(model):
    #model = resnet101(pretrained=True).eval()
    example_inputs = torch.randn(3, 1024,2048)
    # 1. build dependency graph for resnet18
    DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

    # 2. Select some channels to prune. Here we prune the channels indexed by [2, 6, 9].
    pruning_idxs = pruning_idxs=[2, 6, 9]
    pruning_group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=pruning_idxs )

    # 3. prune all grouped layer that is coupled with model.conv1
    if DG.check_pruning_group(pruning_group):
        pruning_group.exec()

    print("After pruning:")
    print(model)

    print(pruning_group)

def magnitude_prune(model):
    model = resnet101(pretrained=True)
    example_inputs = torch.randn(1, 3, 1024,2048)

    # 0. importance criterion for parameter selections
    imp = tp.importance.MagnitudeImportance(p=2, group_reduction='mean')

    # 1. ignore some layers that should not be pruned, e.g., the final classifier layer.
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
            ignored_layers.append(m) # DO NOT prune the final classifier!

            
    # 2. Pruner initialization
    iterative_steps = 5 # You can prune your model to the target sparsity iteratively.
    pruner = tp.pruner.MagnitudePruner(
        model, 
        example_inputs, 
        global_pruning=True, # If False, a uniform sparsity will be assigned to different layers.
        importance=imp, # importance criterion for parameter selection
        iterative_steps=iterative_steps, # the number of iterations to achieve target sparsity
        ch_sparsity=0.5, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        ignored_layers=ignored_layers,
    )

    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    for i in range(iterative_steps):
        # 3. the pruner.step will remove some channels from the model with least importance
        pruner.step()
        
        # 4. Do whatever you like here, such as fintuning
        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        print(model)
        print(model(example_inputs).shape)
        print(
            "  Iter %d/%d, Params: %.2f M => %.2f M"
            % (i+1, iterative_steps, base_nparams / 1e6, nparams / 1e6)
        )
        print(
            "  Iter %d/%d, MACs: %.2f G => %.2f G"
            % (i+1, iterative_steps, base_macs / 1e9, macs / 1e9)
        )
        # finetune your model here
        # finetune(model)
        # ...


def main():
    
    

    torch.backends.cudnn.benchmark = cfg['cudnn']['benchmark']
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    test_kwargs = {'batch_size': cfg['test']['batch_size'],
                   'shuffle': False}  # val and test
    if use_cuda:
        cuda_kwargs = {'num_workers': cfg['workers'],
                       'pin_memory': True}
        test_kwargs.update(cuda_kwargs)

    model_dirname = 'deeplabv3_{0}_{1}_{2}'.format(cfg['model']['backbone'], cfg['dataset']['dataset'], args.cfg)
    model_fname = 'deeplabv3_{0}_{1}_{2}_epoch%d.pth'.format(cfg['model']['backbone'], cfg['dataset']['dataset'], args.cfg)
    model_fpath = os.path.join('output', model_dirname, model_fname)


    dataset = Cityscapes('data/cityscapes',train=False, crop_size=769)
    #crop_size=args.crop_size)

    model = getattr(deeplabv3, 'create_resnet101')(
            device=device,
            num_classes=len(dataset.CLASSES))


    model = model.to(device)

    # Inference
    model = model.eval()

    checkpoint = torch.load('output/deeplabv3_cityscapes_base_2023_02_21-04_46_24_PM/model_best_miou_53-89.pt', map_location=device)


    state_dict = {k[7:]: v for k, v in checkpoint['model'].items() if 'tracked' not in k}

    model.load_state_dict(state_dict)

    #summary(model,(3, 1024,2048))


    #pruning methods
    #depGraph(model)
    magnitude_prune(model)

    




if __name__ == "__main__":
    with open('configs/deeplabv3/deeplabv3_cityscapes_base.yaml', 'r') as file:
        prim_service = yaml.safe_load(file)
    main()
