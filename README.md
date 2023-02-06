# DeepLabv3 Implementation in PyTorch
Implementation of DeepLabv3 semantic segmentation model in PyTorch for ECE-8550 final project. The goal of this project is to implement and improve a research paper. A lot of code is referenced and used from [PyTorch's offical github page](https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py) and an [independent approach from chenxi116](https://github.com/chenxi116/DeepLabv3.pytorch). While my version is very similar to other approaches, I have commented a majority of lines for my own understanding and for others who may be confused. Some comments could be inaccurate as I may have misunderstood some of the code but I believe a majority, if not all, are correct.

After understanding and implementing DeepLabv3, the model was trained and inferenced on the PASCAL VOC 2012 dataset augmented with SBD like in the DeepLabv3 paper. To further test and improve the paper, the model was trained and tested on [RELLIS-3D off-road terrain dataset](https://unmannedlab.github.io/research/RELLIS-3D).

This project includes most aspects of the DeepLabv3 paper such as a ResNet backbone modified for DeepLab and atrous spatial pyramid pooling (ASPP).  Unfortunately, I was not able to implement the multi-grid due to confusion and time constraints for my project. The model still achieves an mIoU score of __73.90%__ on the augmented pascal voc 2012 dataset.

## Setup for a local machine
### Installation
Clone this github repository
```bash
git clone https://github.com/bradford415/deeplabv3-pytorch.git
```

### Anaconda environment
Create a virtual environment with the required dependencies and activate it. Note: there is a bug where conda installs the wrong torchvision version so it needs to be installed after creating the environment using pip.
```bash
conda create -n deeplabv3-pytorch python=3.7 pytorch numpy scipy pillow matplotlib
pip install torchvision
source activate deeplabv3-pytorch
```
### Traning the model
After preparing a dataset, to train the model run the ```train.sh``` script or run ```main.py``` and specify the command line arguments. A brief explanation of the command line arguments are at the top of ```main.py```. Instructions on preparing the augemented pascal dataset are written after the setup.
```bash
bash train.sh
```
or
```bash
python main.py --train --experiment bn_lr7e-3 --backbone resnet101 --dataset pascal --epochs 50 --batch_size 4 --base_lr 0.007
```
A directory is created named after the command line arguments: backbone, experiment, and dataset. The trained model and checkpoints are stored here. Checkpoints are saved at every 10 epoch intervals.

### Testing the model
Once you have a trained model, to test the model run the ```inference.sh``` script or remove the ```--train``` argument from the command line argument. The trained model that is loaded is named by the backbone, experiment. dataset, and epochs.
```
bash inference.sh
```
During inference, the epoch, iteration, and loss will be printed. When inference is finished, the segmented images will be saved, the IoU of each class and the final mIoU for the test set are printed out and saved to a text file. This is all stored in the ```data/<experiment>``` directory.

## Preparing Datasets
### Preparing Rellis-3D camer dataset
First, manually download the camera dataset [Here](https://drive.google.com/file/d/1F3Leu0H_m6aPVpZITragfreO_SGtL2yV/view) and move the .zip folder into ```data/rellis/```. Next, extract ```Rellis_3D_pylon_camera_node.zip```and split the data into train/test (30% test data) set using the ```rellis_split.py``` script with the following commands:
```bash
cd data/rellis # Starting from project root directory
unzip Rellis_3D_pylon_camera_node.zip
cd ../../scripts
python rellis_split.py
```
Finally, using a text editor change the ```--dataset``` command-line arguments in ```train.sh``` and ```inference.sh``` to ```rellis``` 
or on the command-line if not using the ```.sh``` scripts.

### Preparing augmented pascal voc 2012 dataset
First, manually download the augmented dataset [Here](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0) provided by [DrSleep](https://github.com/DrSleep/tensorflow-deeplab-resnet) and move the .zip folder into ```data/pascal/```. It contains ```10582``` images for training and ```1449``` images for validation. Next, finish preparation by extracting the ```SegmentationClassAug.zip``` and downloading the Pascal jpg images with the following commands:
```bash
cd data/pascal # Starting from project root directory
unzip SegmentationClassAug.zip
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_11-May-2012.tar
```

## Visualization
After training a model and running inference, a random image can be visualized using the ```visualize.py``` script. This script grabs a random image from the ```output/<experiement_name>/inference``` directory, adds a legend to it for each class and saves it to a ```.png``` file in the ```output/visualizations``` directory. Currently, this script only supports rellis image visualizations.

__Label a segmented image__

Need to specify ```--label_image``` and the path to the inference folder.
```bash
cd scripts
python visualize.py --label_image --images_path output/example_output/inference
```
label image example output
![label_image](output/visualizations/example_image.png)
