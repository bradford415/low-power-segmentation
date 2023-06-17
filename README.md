# Low Power Semantic Segmentation
TODO: Write the goal of this project

## Table of Contents
TODO fill this out
* [File Structure](#file-structure)
* [Local Setup and Development](#local-setup-and-development)
* [Launching the Docker Container](#launching-the-docker-container)
* [Features](#features)
* [Deploying the Docker Container on the Husky](#deploying-the-docker-container-on-the-husky)
* [Preparing Datsets](#preparing-datasets)
* [Publications](#publications)

## File Structure
TODO: Fill this out

	.
	├── configs                 # Live inference for .onnx files
	├── data                    # Train/test segmentation models (e.g., SwiftNet)
	├── datasets                # DeepLabV3+ with RGB/LiDAR fusion (written by Max in MATLAB)
	├── evaluation              # Helper scripts disconnected from the pipeline
	├── networks                # Archive of files which were never integrated into the pipeline (these should not be used)
	├── quantization	    #
	├── scripts	      	    # Script to run the local docker container
 	├── solution                #
 	├── trained-models	    #
 	├── utils		    #
  	├── environment.yml	    # File to setup the anaconda environment
 	├── compress.sh		    #
 	├── run.pbs.sh		    #
  	├── test.py		    #
	└── train.py 	      	    # 


## Anaconda Environment Setup
```bash
conda env create -f environment.yml
conda activate low-power
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
```

## Running the Project Locally 

### Activate the Conda Environment
```bash
source activate low-power
```

### Train
```bash
python train.py --cfg configs/deeplabv3/deeplabv3_cityscapes_base.yaml
```

### Test
Copy the trained model (```.pt```) into the ```trained-models``` directory and modify the ```model_path``` variable in the config file to the name of the trained model

Run the ```test.py``` by specifying the desired configuartion file (```.yaml```)
```bash
python test.py --cfg configs/deeplabv3/deeplabv3_cityscapes_base.yaml
```

## Preparing the Trained Model for the Jetson Evaluation
Follow these steps once you have a trained model and it is ready for Jetson evaluation.

1. Copy the trained model file ```.pt``` to the ```solution``` dir.
1. Copy the model architecture (in ```networks```) to ```solution/utils```.
1. In ```solution/main.py``` instantiate the model architecture and load the weights in ```main()```. Any additional preprocessing needed should be added to ```loadImageToTensor()``` function.

Once the model is set up in the ```solution``` dir, zip the ```solution``` directory and move it to the ```evaluation``` directory with
```bash
./compress.sh
```

__Before moving the solution to the Jetson, first test it locally to verify correctness.__

Inside ```evaluation```, set up the ```test``` dir with the validation images (normally it would be the test images but they are private for this competition). The ```test``` dir should look as follows:
 	
    	├── test                    
    	│   ├── GT          # Ground truth labeled validation images
    	│   └── IMG         # Raw RGB validation images
    	└── 

Inside the ```evaluation/evaluate.bash``` script, modify the following variables to the correct absolute paths
1. ```path``` to path of ```evaluation``` dir
2. ```testImagesDirectory``` to path of ```evaluation/test/IMG``` dir
3. ```testGroundTruthImagesDirectory``` to path of ```evaluation/test/GT```

TODO: Write instructions on how to run the solution locally

## Running the Project on the Palmetto Super Computer 
### Setting Up the PBS Script
From a login node, create or modify a config file in ```configs``` directory with the desired hyperparameters, dataset, and weight file (if testing). _Be sure to keep a similar naming convention_. Next, modify the ```run.pbs``` file to use the Conda environment, the ```train.py``` or ```test.py``` file, and the config file you just created:
```bash
# Train
python train.py --cfg <configs/cityscapes/config_file.yaml>

# Test
python test.py --cfg <configs/cityscapes/config_file.yaml>
```

### Submitting the Job
The default pbs script is set up to use the viprgs queue with A100 GPUs. If you do not have access to this queue then you will need to remove the queue flag. Submit a train job with the following command:
```bash
qsub run.pbs
```

You can view the job status at any point using ```qstat -u <username>```

### Requesting an Interactive Node
```bash
qsub -I -q viprgs -l select=1:ncpus=20:ngpus=2:mem=128gb:gpu_model=a100,walltime=6:00:00
```

## Inferecing on the Jetson Nano


## Literature Review

### Pruning
- [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710)
  - Structured pruning of convolutional kernels
- [Pruning and Quantization for Deep Neural Network Acceleration: A Survey](https://arxiv.org/abs/2101.09671)
  - Survey on pruning and quantization
- [A Systematic DNN Weight Pruning Framework using Alternating Direction Method of Multipliers](https://arxiv.org/abs/1804.03294)
  - ADMM pruning

### Quantization
- [Fundamentals of Quantization](https://pytorch.org/blog/quantization-in-practice/#fundamentals-of-quantization)
- [Quantizing deep convolutional networks for efficient inference: A whitepaper
](https://arxiv.org/abs/1806.08342)
  - Recommended quantization protocols (cited a lot by PyTorch)
- [Static Quantization with Eager Mode in PyTorch](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)
  - Example for PyTorch Post-training and Quantization-aware training (QAT)
