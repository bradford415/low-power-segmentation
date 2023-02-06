"""Main file to run this project
"""
import argparse
import os
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
import deeplabv3
from utils import AverageMeter, inter_and_union
from datasets.pascal import VOCSegmentation
from datasets.cityscapes import Cityscapes
from datasets.rellis import Rellis3D


parser = argparse.ArgumentParser()
# If '--train' is present in the command line, args.train will equal True
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--experiment', type=str, required=True,
                    help='name of experiment')
parser.add_argument('--backbone', type=str, default='resnet101',
                    help='resnet101')
parser.add_argument('--dataset', type=str, default='pascal',
                    help='pascal or cityscapes')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of training iterations')
parser.add_argument('--batch_size', type=int, default=16,
                    help='number of samples to train at a time per iteration')
parser.add_argument('--base_lr', type=float, default=16,
                    help='base learning rate')
parser.add_argument('--last_mult', type=float, default=1.0,
                    help='learning rate multiplier for last layers')
parser.add_argument('--scratch', action='store_true', default=False,
                    help='train from scratch, do not use pre-trained weights')
# pascal: training => 513x513, testing => 513x513
# cityscapes: training => 769x769, testing => 1025x2049
# rellis: training => 721x721, testing => 1201x1921
# Deeplab framework crop size explained here: https://github.com/google-research/deeplab2/blob/main/g3doc/faq.md
parser.add_argument('--crop_size', type=int, default=513,
                    help='image crop size')  # default value for pascal VOC dataset
parser.add_argument('--resume', type=str, default=None,
                    help='path to checkpoint to resume from')
args = parser.parse_args()


def main():
    # cudnn.benchmark finds best algorithm and can speed up runtime
    # Only use if your input size does not vary and
    # your model stays the same for every interation
    torch.backends.cudnn.benchmark = True
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': args.batch_size,
                   'shuffle': False}  # val and test
    if use_cuda:
        cuda_kwargs = {'num_workers': 2,
                       'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

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
    if args.dataset == 'pascal':
        dataset = VOCSegmentation('data/pascal',
                                  train=args.train, crop_size=513)#crop_size=args.crop_size)
    elif args.dataset == 'cityscapes':
        dataset = Cityscapes('data/cityscapes',
                             train=args.train, crop_size=769)#crop_size=args.crop_size)
    elif args.dataset == 'rellis':
        dataset = Rellis3D('data/rellis',
                             train=args.train, crop_size=721)#crop_size=args.crop_size)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    # In this case, getattr() is calling a function from deeplab.py file to return the model
    # and the following parenthesis pass arguments to this 'resnet101' function
    # I am not sure the advantage over this rather than just calling the function itself
    # w/o getattr()
    if args.backbone == 'resnet101':
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
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        # only using gpu:0 (NVIDIA TITAN RTX) bc it is much more powerful than
        # my gpu:1 (NVIDIA GeForce RTX 2060) and that causes gpu:1 to be a bottle neck
        if use_cuda:
            model = nn.DataParallel(model, device_ids=[0])

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
        optimizer = optim.SGD(params_to_optimize, lr=args.base_lr,
                            momentum=0.9, weight_decay=0.0001)
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

        start_epoch = 0  # defined if loading a checkpoint and need to start a specific epoch
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

    else:  # Inference
        model = model.eval()  # Required to set BN layers to eval mode
        print('=> loading checkpoint {0}'.format(model_fpath.split('/')[-1] % args.epochs))
        checkpoint = torch.load(model_fpath % args.epochs, map_location=device)
        print('=> loaded checkpoint {0} (epoch {1})'.format(
                    model_fpath.split('/')[-1] % args.epochs, checkpoint['epoch']))
        # Because the model was trained with nn.DataParallel each layer is wrapped
        # in a .module(). We are not inferencing with DataParallel so we have
        # to remove the 'module.' in front of each layer or else it will not
        # be able to find those layers. Starting the key name at element 7
        # will remove this 'module.' This is what the following line does.
        state_dict = {k[7:]: v for k,
                    v in checkpoint['model'].items() if 'tracked' not in k}
        # Do not need to load optimizer state_dict because it is not used for inference
        model.load_state_dict(state_dict)
        if args.dataset == 'pascal':
            cmap = loadmat('data/pascal/pascal_seg_colormap.mat')['colormap']
            cmap = (cmap * 255).astype(np.uint8).flatten().tolist()
        elif args.dataset == 'rellis':
            cmap = np.array(dataset.color_map).flatten().tolist()
        else:
            raise ValueError(
                'Unknown colormap for dataset: {}'.format(args.dataset))

        dataset_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)

        inter_meter = AverageMeter()  # metrics for intersection
        union_meter = AverageMeter()  # metrics for union
        with torch.inference_mode():  # newer torch.no_grad()
            # Running count of the number of images inferred so far.
            # Used to index into the masks (ground truths) of the dataset.
            mask_index = 0
            for index, (data, target) in enumerate(dataset_loader):
                data, target = data.to(device), target.to(device)
                # model(data.unsqueeze(0)) needs to be added if you loop through
                # dataset instead of dataset_loader, this is to add the batch dimension
                # [B, C, H, W]
                outputs = model(data)  # model(data).unsqueeze(0)
                _, pred = torch.max(outputs, 1)
                pred = pred.cpu().data.numpy().squeeze().astype(np.uint8)
                # move data back to cpu to use numpy
                mask = target.cpu().numpy().astype(np.uint8)
                # Need to interpret 1 image at a time in order to work with some PIL functions
                # Alternitavely, you could loop through len(dataset) and process 1 sample at a time
                # not using the dataloader at all and it would get rid of the following loop.
                # However, then you would only be using a batch size of 1 but probably doesn't matter.
                for mask_index, (image_pred, image_mask) in enumerate(zip(pred, mask), mask_index):
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
                    # If on the last iteration of the loop, increment mask_index.
                    # This is necessary bc when the for loop restarts for the new batch
                    # it will not increment and we want to keep track of every image
                    if mask_index % args.batch_size == args.batch_size - 1:
                        mask_index += 1

            iou = inter_meter.sum / (union_meter.sum + 1e-10)
            # Print and save IoU per class and final mIoU score
            with open(os.path.join(model_path, 'metrics.txt'), 'w') as file:
                for i, val in enumerate(iou):
                    print('IoU {0}: {1:.2f}'.format(dataset.CLASSES[i], val * 100))
                    file.write('IoU {0}: {1:.2f}\n'.format(
                        dataset.CLASSES[i], val * 100))
                print('Mean IoU: {0:.2f}'.format(iou.mean() * 100))
                file.write('Mean IoU: {0:.2f}'.format(iou.mean() * 100))


if __name__ == '__main__':
    main()
