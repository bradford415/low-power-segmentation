"""train.py

File to train segmentation models
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pdb 
import numpy as np
from scipy.io import loadmat
from PIL import Image
from pathlib import Path

# Import local files
import deeplabv3
from utils import AverageMeter, inter_and_union
from pascal import VOCSegmentation
from cityscapes import Cityscapes
from rellis import Rellis3D


parser = argparse.ArgumentParser()
# If '--train' is present in the command line, args.train will equal True
parser.add_argument('--cfg', type=str, required=True)
parser.add_argument('--experiment', type=str, required=True,
                    help='name of experiment')
args = parser.parse_args()

with open(args.cfg, "r") as stream:
    try:
        cfg = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


def train(config):
    torch.backends.cudnn.benchmark = cfg['cudnn']['benchmark']
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    train_kwargs = {'batch_size': cfg['train']['batch_size'], 'shuffle': cfg['train']['shuffle']}

    if use_cuda:
        cuda_kwargs = {'num_workers': cfg['workers'],
                       'pin_memory': True} 
                       # pin speeds up host to device tensor transfer
                       # (should always be true when using a nvidia gpu)
        train_kwargs.update(cuda_kwargs)

    # File and dir name to save checkpoints
    # Model checkpoints are saved w/ this naming convention during training
    model_dirname = 'deeplabv3_{0}_{1}_{2}'.format(
        args.backbone, args.dataset, args.experiment)
    model_fname = 'deeplabv3_{0}_{1}_{2}_epoch%d.pth'.format(
        args.backbone, args.dataset, args.experiment)
    model_path = os.path.join('output', model_dirname)
    model_fpath = os.path.join('output', model_dirname, model_fname)
    Path(model_path).mkdir(parents=True, exist_ok=True)

    # Crop size is currently hard coded but can be changed to use args.crop_size
    if cfg['dataset']['dataset'] == 'pascal':
        dataset = VOCSegmentation('data/pascal',
                                  train=args.train, crop_size=513)#crop_size=args.crop_size)
    elif cfg['dataset']['dataset'] == 'cityscapes':
        dataset = Cityscapes('data/cityscapes',
                             train=args.train, crop_size=769)#crop_size=args.crop_size)
    elif cfg['dataset']['dataset'] == 'rellis':
        dataset = Rellis3D('data/rellis',
                             train=args.train, crop_size=721)#crop_size=args.crop_size)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    # In this case, getattr() is calling a function from deeplab.py file to return the model
    # and the following parenthesis pass arguments to this 'resnet101' function
    # I am not sure the advantage over this rather than just calling the function itself
    # w/o getattr()
    if cfg['model']['backbone'] == 'resnet101':
        model = getattr(deeplabv3, 'create_resnet101')(
            pretrained=(not args.scratch),
            device=device,
            num_classes=len(dataset.CLASSES))
    else:
        raise ValueError('Unknown backbone: {}'.format(args.backbone))

    model = model.to(device)

    """Notes:
    - ignore_index ignores the 255 value bc the augmented dataset labels uses 255 
      (white) as the border around the objects and we want to ignore for training
    - DataParallel splits input across devices. During forward pass the model is 
      replicated on each device and each device handles a portion of the input.
    - .cuda(), the same as, .to(device) used to put models/tensors on gpu 
      .to(device) is more flexible and should probably be used more
    """
    if args.train:
        model.train()
        criterion = nn.CrossEntropyLoss(ignore_index=cfg['train']['ignore_label'])
        # only using gpu:0 (NVIDIA TITAN RTX) bc it is much more powerful than
        # my gpu:1 (NVIDIA GeForce RTX 2060) and that causes gpu:1 to be a bottle neck
        if use_cuda:
            model = nn.DataParallel(model, device_ids=cfg['gpus'])

        # PyTorch grabs the optimization parameters slightly different shown here:
        # https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/_utils.py
        # Their backbone and and classifier is defined here:
        # https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py
        # Where class DeepLabV3 uses inheritance defined here:
        # https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/_utils.py
        # I am also not sure why you cannot just use model.parameters() in optim.SGD()

        # Pull layer parameters from ResNet class in deeplabv3.py
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
        if cfg['optimizer']:
            optimizer = optim.SGD(params_to_optimize, lr=cfg['train']['base_lr'],
                                momentum=cfg['train']['momentum'], weight_decay=cfg['train']['weight_decay'])
        losses = AverageMeter()
        dataset_loader = torch.utils.data.DataLoader(dataset, **train_kwargs)

        max_iterations = args.epochs * len(dataset_loader)

        # Resume not tested yet
        if args.resume:
            if os.path.isfile(args.resume):
                print('=> loading checkpoint {0}'.format(args.resume))
                checkpoint = torch.load(args.resume)
                start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print('=> loaded checkpoint {0} (epoch {1})'.format(
                    args.resume, checkpoint['epoch']))
            else:
                print('=> no checkpoint found at {0}'.format(args.resume))

        start_epoch = configs['train']['start_epoch']
        end_epoch = configs['train']['end_epoch']
        for epoch in range(start_epoch, args.epochs):
            for index, (data, target) in enumerate(dataset_loader):
                current_iteration = epoch * len(dataset_loader) + index
                # Learning rate updated based on Deeplabv3 paper section 4.1
                # Uses a 'poly' learning rate policy
                # max_iterations is defined as (num_epochs*num_iterations_per_epoch)
                # num_iterations_per_epoch I think is ceil(num_training_samples/batch_size)
                # same as len(dataset_loader)
                lr = args.base_lr * \
                    (1 - float(current_iteration) / max_iterations) ** 0.9
                optimizer.param_groups[0]['lr'] = lr  # Update lr for backbone
                # Update lr for ASPP, I think thats what [1] means
                optimizer.param_groups[1]['lr'] = lr * args.last_mult

                # Put tensors on a gpu
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, target)
                if np.isnan(loss.item()) or np.isinf(loss.item()):
                    pdb.set_trace()
                # Keep track of running loss
                losses.update(loss.item(), args.batch_size)

                loss.backward()
                optimizer.step()

                print('epoch: {0}\t'
                    'iter: {1}/{2}\t'
                    'lr: {3:.6f}\t'
                    'loss: {loss.val:.4f} ({loss.ema:.4f})'.format(
                        epoch + 1, index + 1, len(dataset_loader), lr, loss=losses))

            # Save a checkpoint every 10 epochs
            if epoch % 10 == 9:
                torch.save({
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, model_fpath % (epoch + 1))


if __name__ == '__main__':
    main()
