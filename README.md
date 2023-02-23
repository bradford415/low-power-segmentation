# Low Power Semantic Segmentation

## Running the Model Locally 
### Train
```bash
python train.py --cfg configs/deeplabv3/deeplabv3_cityscapes_base.yaml
```

### Test
```bash
python test.py --cfg configs/deeplabv3/deeplabv3_cityscapes_base.yaml
```

## Running the Model on the Palmetto Super Computer 
### Train
```bash
```

### Test
```bash
```




## Literature Review

### Pruning
- [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710)
  - Structured pruning of convolutional kernels
