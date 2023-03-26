# Low Power Semantic Segmentation

## Anaconda Environment Setup
```bash
conda env create -f environment.yml
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
```bash
python test.py --cfg configs/deeplabv3/deeplabv3_cityscapes_base.yaml
```

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

## Literature Review

### Pruning
- [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710)
  - Structured pruning of convolutional kernels
- [Pruning and Quantization for Deep Neural Network Acceleration: A Survey](https://arxiv.org/abs/2101.09671)
  - Survey on pruning and quantization
- [A Systematic DNN Weight Pruning Framework using Alternating Direction Method of Multipliers](https://arxiv.org/abs/1804.03294)
  - ADMM pruning
