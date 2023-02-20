"""eval.py

File to test and evaluate segmentation models
"""
import argparse
import os
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


def test():
    torch.backends.cudnn.benchmark = cfg['cudnn']['benchmark']
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    test_kwargs = {'batch_size': cfg['test']['batch_size'],
                   'shuffle': False}  # val and test
    if use_cuda:
        cuda_kwargs = {'num_workers': cfg['workers'],
                       'pin_memory': True}
        test_kwargs.update(cuda_kwargs)

    # File and dir name to save checkpoints
    # Model checkpoints are saved w/ this naming convention during training
    model_dirname = 'deeplabv3_{0}_{1}_{2}'.format(
        cfg['model']['backbone'], cfg['dataset']['dataset'], args.cfg)
    model_fname = 'deeplabv3_{0}_{1}_{2}_epoch%d.pth'.format(
        cfg['model']['backbone'], cfg['dataset']['dataset'], args.cfg)
    model_path = os.path.join('output', model_dirname)
    model_fpath = os.path.join('output', model_dirname, model_fname)
    Path(model_path).mkdir(parents=True, exist_ok=True)

    # Crop size is currently hard coded but can be changed to use args.crop_size
    if cfg['dataset']['dataset'] == 'pascal':
        dataset = VOCSegmentation('data/pascal',
                                  train=False, crop_size=513)#crop_size=args.crop_size)
    elif cfg['dataset']['dataset'] == 'cityscapes':
        dataset = Cityscapes('data/cityscapes',
                             train=False, crop_size=769)#crop_size=args.crop_size)
    elif cfg['dataset']['dataset'] == 'rellis':
        dataset = Rellis3D('data/rellis',
                             train=False, crop_size=721)#crop_size=args.crop_size)
    else:
        raise ValueError('Unknown dataset: {}'.format(cfg['dataset']['dataset']))

    if cfg['model']['backbone'] == 'resnet101':
        model = getattr(deeplabv3, 'create_resnet101')(
            pretrained=(not args.scratch),
            device=device,
            num_classes=len(dataset.CLASSES))
    else:
        raise ValueError('Unknown backbone: {}'.format(cfg['model']['backbone']))

    model = model.to(device)

    # Inference
    model = model.eval()  # Required to set BN layers to eval mode
    print('=> loading checkpoint {0}'.format(model_fpath.split('/')[-1] % args.epochs))
    checkpoint = torch.load(model_fpath % args.epochs, map_location=device)
    print('=> loaded checkpoint {0} (epoch {1})'.format(
                model_fpath.split('/')[-1] % args.epochs, checkpoint['epoch']))
    # Remove 'module.' appended by DataParallel() 
    state_dict = {k[7:]: v for k,
                v in checkpoint['model'].items() if 'tracked' not in k}
    # Do not need to load optimizer state_dict because it is not used for inference
    model.load_state_dict(state_dict)
    if cfg['dataset']['dataset'] == 'pascal':
        cmap = loadmat('data/pascal/pascal_seg_colormap.mat')['colormap']
        cmap = (cmap * 255).astype(np.uint8).flatten().tolist()
    elif cfg['dataset']['dataset'] == 'rellis':
        cmap = np.array(dataset.color_map).flatten().tolist()
    else:
        raise ValueError(
            'Unknown colormap for dataset: {}'.format(cfg['dataset']['dataset']))

    dataset_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)

    inter_meter = AverageMeter()  
    union_meter = AverageMeter()  
    with torch.inference_mode():  # newer torch.no_grad()
        mask_index = 0
        for index, (data, target) in enumerate(dataset_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, pred = torch.max(outputs, 1)
            
            pred, target= pred.cpu(), target.cpu()
            if cfg['test']['save_images']:
                colorize(pred, mask, cfg['dataset']['dataset'])
            pred = pred.cpu().data.numpy().squeeze().astype(np.uint8)
            mask = target.cpu().numpy().astype(np.uint8)
            image_name = dataset.masks[mask_index].split('/')[-1]
            mask_pred = Image.fromarray(image_pred)
            mask_pred.putpalette(cmap)

            Path(os.path.join(model_path, 'inference')).mkdir(
                parents=True, exist_ok=True)
            mask_pred.save(os.path.join(
                model_path, 'inference', image_name))
            print('eval: {0}/{1}'.format(mask_index + 1, len(dataset)))
            inter, union = inter_and_union(
                image_pred, image_mask, len(dataset.CLASSES))
            # Keep running sum of intersection and union values of image
            # Inter and union are based on the prediction and groud truth mask
            inter_meter.update(inter)
            union_meter.update(union)

        iou = inter_meter.sum / (union_meter.sum + 1e-10) # 1e-10 is used to prevent division by 0 I think
        # Print and save IoU per class and final mIoU score
        with open(os.path.join(model_path, 'metrics.txt'), 'w') as file:
            for i, val in enumerate(iou):
                print('IoU {0}: {1:.2f}'.format(dataset.CLASSES[i], val * 100))
                file.write('IoU {0}: {1:.2f}\n'.format(
                    dataset.CLASSES[i], val * 100))
            print('Mean IoU: {0:.2f}'.format(iou.mean() * 100))
            file.write('Mean IoU: {0:.2f}'.format(iou.mean() * 100))


def main():

    arg

    with open('configs/base_experiment.yaml', 'r') as file:
        prim_service = yaml.safe_load(file)

    test()


if __name__ == '__main__':
    main()
