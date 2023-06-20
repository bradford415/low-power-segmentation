import torch

num_devices = 1

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

if use_cuda:
    print(f"Using {num_devices} GPU(s): ")
    for gpu in range(num_devices):
        print(f"    -{torch.cuda.get_device_name(gpu)}")
else:
    print('GPU not available')
