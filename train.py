"""train.py

File to train segmentation models
"""
import argparse
import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pdb 
from tqdm import tqdm
import numpy as np
#from scipy.io import loadmat
import shutil
from PIL import Image
from pathlib import Path
from datetime import datetime


# Import local files
from networks import deeplabv3
from utils import AverageMeter, inter_and_union
from datasets import VOCSegmentation
from datasets import Cityscapes
from datasets import Rellis3D


parser = argparse.ArgumentParser()
# If '--train' is present in the command line, args.train will equal True
parser.add_argument('--cfg', type=str, required=True)
parser.add_argument('--experiment', type=str, required=False,
                    help='name of experiment')
args = parser.parse_args()

with open(args.cfg, "r") as stream:
    try:
        cfg = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


def train():
    torch.backends.cudnn.benchmark = cfg['cudnn']['benchmark']
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    train_kwargs = {'batch_size': cfg['train']['batch_size'], 'shuffle': cfg['train']['shuffle']}
    val_kwargs = {'batch_size': cfg['test']['batch_size'], 'shuffle': ['train']['shuffle']}  # val and test
    
    if use_cuda:
        print(f"\nUsing GPU(s): {torch.cuda.get_device_name(cfg['gpus'])}\n")
        cuda_kwargs = {'num_workers': cfg['workers'],
                       'pin_memory': True} 
                       # pin speeds up host to device tensor transfer
                       # (should always be true when using a nvidia gpu)
        train_kwargs.update(cuda_kwargs)
        val_kwargs.update(cuda_kwargs)
    else:
        print('Using CPU')

    # File and dir name to save checkpoints
    # Model checkpoints are saved w/ this naming convention during training
    model_dirname = f"{args.cfg}_{datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')}"
    model_fname = 'model'
    model_path = os.path.join('output', model_dirname)
    model_fpath = os.path.join('output', model_dirname, model_fname)
    Path(model_path).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.cfg, model_fpath)

    # Create Train and validation dataset (validation to test accuracy after every epoch)
    if cfg['dataset']['dataset'] == 'pascal':
        dataset_train = VOCSegmentation('data/pascal',
                                        train=False, crop_size=513)
        dataset_val = VOCSegmentation('data/pascal',
                                      train=False)
    elif cfg['dataset']['dataset'] == 'cityscapes':
        dataset_train = Cityscapes(cfg['dataset']['root'],
                                   train=False)
        dataset_val = Cityscapes(cfg['dataset']['root'],
                                 train=False)
    elif cfg['dataset']['dataset'] == 'rellis':
        dataset_train = Rellis3D('data/rellis',
                           train=True, crop_size=721)
        dataset_val = Rellis3D('data/rellis',
                           train=False)
    else:
        raise ValueError('Unknown dataset: {}'.format(cfg['dataset']['dataset']))

    # In this case, getattr() is calling a function from deeplab.py file to return the model
    # and the following parenthesis pass arguments to this 'resnet101' function
    # I am not sure the advantage over this rather than just calling the function itself
    # w/o getattr()
    if cfg['model']['backbone'] == 'resnet101':
        model = getattr(deeplabv3, 'create_resnet101')(
            pretrained=(not False),
            device=device,
            num_classes=len(dataset_train.CLASSES))
    else:
        raise ValueError('Unknown backbone: {}'.format(cfg['model']['backbone']))

    model = model.to(device)

    """Notes:
    - for pascal dataset the ignore_index ignores the 255 value bc the augmented 
      dataset labels uses 255 (white) as the border around the objects and we want 
      to ignore for training
    - DataParallel splits input across devices. During forward pass the model is 
      replicated on each device and each device handles a portion of the input.
    - .cuda(), the same as, .to(device) used to put models/tensors on gpu 
      .to(device) is more flexible and should probably be used more
    """
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=cfg['train']['ignore_label'])
    if use_cuda:
        model = nn.DataParallel(model, device_ids=cfg['gpus'])

    # PyTorch grabs the optimization parameters slightly different shown here:
    # https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/_utils.py
    # Their backbone and and classifier is defined here:
    # https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py
    # Where class DeepLabV3 uses inheritance defined here:
    # https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/_utils.py

    # separate backbone and aspp parameters to set different learning rates
    backbone_params = (
        list(model.module.conv1.parameters()) +
        list(model.module.bn1.parameters()) +
        list(model.module.layer1.parameters()) +
        list(model.module.layer2.parameters()) +
        list(model.module.layer3.parameters()) +
        list(model.module.layer4.parameters())
    )
    aspp_params = list(model.module.aspp.parameters())
    # Create a list of dictionaries to store the backbone and the aspp parameters
    # Optimize only the trainable parameters ('requires_grad')
    params_to_optimize = [
        {'params': filter(lambda p: p.requires_grad, backbone_params)},
        {'params': filter(lambda p: p.requires_grad, aspp_params)}
    ]
    if cfg['train']['optimizer'] == 'sgd':
        optimizer = optim.SGD(params_to_optimize, lr=cfg['train']['base_lr'],
                            momentum=cfg['train']['momentum'], weight_decay=cfg['train']['weight_decay'])
    losses = AverageMeter()
    loader_train = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
    loader_val = torch.utils.data.DataLoader(dataset_val, **val_kwargs)

    max_iterations = cfg['train']['end_epoch'] * len(loader_train)

    # Resume not tested yet
    if cfg['train']['resume']:
        if os.path.isfile(cfg['train']['resume']):
            print('=> loading checkpoint {0}'.format(cfg['train']['resume']))
            checkpoint = torch.load(cfg['train']['resume'])
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint {0} (epoch {1})'.format(
                cfg['train']['resume'], checkpoint['epoch']))
        else:
            print('=> no checkpoint found at {0}'.format(cfg['train']['resume']))

    print('Training...\n')
    start_epoch = cfg['train']['start_epoch']
    end_epoch = cfg['train']['end_epoch']
    for epoch in range(start_epoch, end_epoch):
        print(f'Epoch {epoch+1}: ')
        model.train()
        for index, (data, target) in enumerate(tqdm(loader_train, ascii=' >=')):
            current_iteration = epoch * len(loader_train) + index
            # Learning rate updated based on Deeplabv3 paper section 4.1 => 'poly' learning rate
            lr = cfg['train']['base_lr'] * \
                (1 - float(current_iteration) / max_iterations) ** 0.9
            # Update lr for [0] backbone and [1] aspp
            optimizer.param_groups[0]['lr'] = lr  
            optimizer.param_groups[1]['lr'] = lr * cfg['train']['last_mult'] # LR multiplier for last layers

            # Put tensors on a gpu
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            if np.isnan(loss.item()) or np.isinf(loss.item()):
                pdb.set_trace()
            # Keep track of running loss
            losses.update(loss.item(), cfg['train']['batch_size'])

            loss.backward()
            optimizer.step()

        # Evaluate epoch
        model.eval()
        inter_meter = AverageMeter()  
        union_meter = AverageMeter()
        print('Evaluating on validation set...')
        with torch.inference_mode:
            for index, (data, target) in enumerate(loader_val):
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, pred = torch.max(outputs, 1)
                # move data back to cpu to use numpy
                Path(os.path.join(model_path, 'inference')).mkdir(
                    parents=True, exist_ok=True)
                inter, union = inter_and_union(
                    pred, target, len(dataset_train.CLASSES))
                # Keep running sum of intersection and union values of image
                # Inter and union are based on the prediction and groud truth mask
                inter_meter.update(inter)
                union_meter.update(union)

        print(f'epoch: {epoch+1}\t average loss: {losses.avg}')
        # Save a checkpoint every 10 epochs
        if epoch % 10 == 9:
            torch.save({
                'epoch': epoch + 1, # +1 because when loading a checkpoint you want to start at the next epoch
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, model_fpath % (epoch + 1))
    



if __name__ == '__main__':
    train()
