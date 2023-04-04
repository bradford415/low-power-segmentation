"""Dataset file for the low power challenge. Used to preprocess the data and 
prepare it for the data loader. The lpcv dataset can be found here:
https://lpcv.ai/2023LPCVC/program

Notes:

"""
from torch.utils.data import Dataset
import os
from pathlib import Path
import numpy as np
from utils import preprocess
from PIL import Image
import glob

class lpcvc(Dataset):
    CLASSES = [
        'background', 'avalanche', 'building_undamaged', 'building_damaged', 'cracks/fissure/subsidence', 'debris/mud/rock_flow', 'fire/flare', 'flood/water/river/sea',
        'ice_jam_flow', 'lava_flow', 'person', 'pyroclastic_flow', 'road/railway/bridge', 'vehicle'
    ]

    def __init__(self, root, train=True, transform=None, target_transform=None, crop_size=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.crop_size = crop_size

        dataset_split = 'LPCVC_Train' if self.train else 'LPCVC_Val'
        self.images = self._get_files(dataset_split, 'IMG')[:100]
        self.masks = self._get_files(dataset_split, 'GT')[:100]
        self.num_samples = len(self.images)
        assert len(self.images) == len(self.masks)


    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.masks[index]).convert('L')

        _img, _target = preprocess(_img,
                                _target,
                                flip=True if self.train else False,
                                scale=(0.5, 2.0) if self.train else None,
                                crop=(self.crop_size, self.crop_size) if self.train else (512, 512))
        
        
        if self.transform is not None:
            _img = self.transform(_img)

        if self.target_transform is not None:
            _target = self.target_transform(_target)
        return _img, _target
    
    def _get_files(self, dataset_split, data_type):
        dataset_path = os.path.join(self.root, dataset_split, data_type)
        filenames = glob.glob(f'{self.root}/{dataset_split}/{data_type}/*')
        
        return sorted(filenames)
        
    def __len__(self):
        return len(self.images)
