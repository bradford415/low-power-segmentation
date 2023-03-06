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
from PIL import Image
from pathlib import Path
from datetime import datetime

# Import local files
from networks import deeplabv3
from utils import AverageMeter, inter_and_union
from utils import colorize, color_maps
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
                   'shuffle': False} 
    if use_cuda:
        # Set the single GPU we want to use
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg['gpus'][0]) 
        print(f"Using GPU: {torch.cuda.get_device_name(cfg['gpus'][0])}")
        cuda_kwargs = {'num_workers': cfg['workers'],
                       'pin_memory': True}
        test_kwargs.update(cuda_kwargs)


    model_weights = cfg['test']['model_path']
    config_name = args.cfg.split('/')[-1].split('.')[0]
    model_dirname = f"{config_name}_{datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')}"
    model_fname = 'model'
    model_path = os.path.join('output', 'inference', model_dirname)
    model_fpath = os.path.join('output', 'inference', model_dirname, model_fname)
    Path(model_path).mkdir(parents=True, exist_ok=True)

    if cfg['test']['save_images']:
        image_path = os.path.join('output', 'inference', model_dirname, 'images')
        Path(image_path).mkdir(parents=True, exist_ok=True)

    cmap = np.array(color_maps[cfg['dataset']['dataset']]).flatten().tolist()
    if cfg['dataset']['dataset'] == 'pascal':
        dataset_test = VOCSegmentation('data/pascal',
                                  train=False, crop_size=None)#crop_size=args.crop_size)
    elif cfg['dataset']['dataset'] == 'cityscapes':
        dataset_test = Cityscapes('data/cityscapes',
                             train=False, crop_size=None)#crop_size=args.crop_size)
        
    elif cfg['dataset']['dataset'] == 'rellis':
        dataset_test = Rellis3D('data/rellis',
                             train=False, crop_size=None)#crop_size=args.crop_size)
    else:
        raise ValueError('Unknown dataset: {}'.format(cfg['dataset']['dataset']))

    if cfg['model']['backbone'] == 'resnet101':
        model = getattr(deeplabv3, 'create_resnet101')(
            pretrained=False,
            device=device,
            num_classes=len(dataset_test.CLASSES))
    else:
        raise ValueError('Unknown backbone: {}'.format(cfg['model']['backbone']))

    # No DataParallel during inference because batch_size = 1
    model = model.to(device)

    print(f'\nNumber of train samples: {dataset_test.num_samples}')

    # Inference
    model = model.eval()  # Required to set BN layers to eval mode
    print(f"Loading model {model_weights}")
    weights = torch.load(model_weights, map_location=device)

    # Remove 'module.' appended by DataParallel() 
    state_dict = {k[7:]: v for k,
                v in weights['model'].items()}
    # Do not need to load optimizer state_dict because it is not used for inference
    model.load_state_dict(state_dict)

    dataset_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    inter_meter = AverageMeter()  
    union_meter = AverageMeter() 
    with torch.inference_mode():  # newer torch.no_grad()
        for index, (data, target) in enumerate(dataset_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, pred = torch.max(outputs, 1)
            
            pred, target= pred.cpu(), target.cpu()
            if cfg['test']['save_images']:
                image_name = os.path.join(image_path,
                    dataset_test.masks[index].split('/')[-1]) 
                colorize(pred, image_name, cmap)
            print('eval: {0}/{1}'.format(index + 1, len(dataset_test)))
            inter, union = inter_and_union(
                pred, target, len(dataset_test.CLASSES))
            # Keep running sum of intersection and union values of image
            # Inter and union are based on the prediction and groud truth mask
            inter_meter.update(inter)
            union_meter.update(union)

        iou = inter_meter.sum / (union_meter.sum + 1e-10) # 1e-10 is used to prevent division by 0 I think
        # Print and save IoU per class and final mIoU score
        with open(os.path.join(model_path, 'metrics.txt'), 'w') as file:
            for i, val in enumerate(iou):
                print('IoU {0}: {1:.2f}'.format(dataset_test.CLASSES[i], val * 100))
                file.write('IoU {0}: {1:.2f}\n'.format(
                    dataset_test.CLASSES[i], val * 100))
            print('Mean IoU: {0:.2f}'.format(iou.mean() * 100))
            file.write('Mean IoU: {0:.2f}'.format(iou.mean() * 100))


def main():
    test()


if __name__ == '__main__':
    main()
