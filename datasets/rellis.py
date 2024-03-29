"""Dataset file for the Rellis-3D. Used to preprocess the data and 
prepare it for the data loader. The Rellis dataset can be found here:
https://unmannedlab.github.io/research/RELLIS-3D

Notes:

"""
from torch.utils.data import Dataset
import os
from pathlib import Path
import numpy as np
from utils import preprocess
from PIL import Image
import glob

class Rellis3D(Dataset):
    CLASSES = [
        'void', 'grass', 'tree', 'pole', 'water', 'sky', 'vehicle', 'object',
        'asphalt', 'building', 'log', 'person', 'fence', 'bush', 'concrete',
        'barrier', 'puddle', 'mud', 'rubble'
    ]

    def __init__(self, root, train=True, crop_size=None):
        self.root = root
        self.train = train
        self.crop_size = crop_size

        dataset_split = 'train' if self.train else 'test'
        self.images = self._get_files(dataset_split, 'rgb')
        self.masks = self._get_files(dataset_split, 'id')
        self.num_samples = len(self.images)
        assert len(self.images) == len(self.masks)

        self.color_map = [
            [0, 0, 0], # void
            #[108, 64, 20], # dirt
            [0, 102, 0], # grass
            [0, 255, 0], # tree
            [0, 153, 153], # pole
            [0, 128, 255], # water
            [0, 0, 255], # sky
            [255, 255, 0], # vehicle
            [255, 0, 127], # object
            [64, 64, 64], # asphalt
            [255, 0, 0], # building
            [102, 0, 0], # log
            [204, 153, 255], # person
            [102, 0, 204], # fence
            [255, 153, 204], # bush
            [170, 170, 170], # concrete
            [41, 121, 255], # barrier
            [134, 255, 239], # puddle
            [99, 66, 34], # mud
            [110, 22, 138]] # rubble
        
        # Used to map the current id labels (masks) to the right class.
        self.label_mapping = {0: 0, 
                              1: 0,
                              3: 1,
                              4: 2,
                              5: 3,
                              6: 4,
                              7: 5,
                              8: 6,
                              9: 7,
                              10: 8,
                              12: 9,
                              15: 10,
                              17: 11,
                              18: 12,
                              19: 13,
                              23: 14,
                              27: 15,
                              29: 1,
                              30: 1,
                              31: 16,
                              32: 4,
                              33: 17,
                              34: 18}
        self.num_classes = len(self.CLASSES)

    def convert_label(self, label, inverse=False):
        """Transform mask labels to class values 0-34 => 0-18
           
        Args
            label: The ground truth image mask to be transformed
            inverse: Bool variable to swap the label_mapping to transform image
                     from class ids to original label values (currently not used)
        """
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp==k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp==k] = v
        return label

    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.masks[index])

        # Convert PIL Image to np array in order to convert the label, then convert back to PIL Image
        _target = np.array(_target)        
        _target = self.convert_label(_target)
        _target = Image.fromarray(_target)
        _img, _target = preprocess(_img,
                                _target,
                                flip=True if self.train else False,
                                scale=(0.5, 2.0) if self.train else None,
                                crop=(self.crop_size, self.crop_size) if self.train else (1201, 1921))

        return _img, _target
    
    def _get_files(self, dataset_split, data_type):
        dataset_path = os.path.join(self.root, 'Rellis-3D-camera-split', dataset_split, data_type)
        #filenames = list(Path(dataset_path).rglob('*.*'))
        filenames = glob.glob(os.path.join(dataset_path,'*.*'))
        return sorted(filenames)
        
    def __len__(self):
        return len(self.images)
