"""Script to make train and test sets for the RELLIS-3D dataset.
Images that have corresponding id images (labels), are you used
for train/test. The split percentage is defined below.
Images that do not have an id image are categorized as unlabeled.
Maybe could look into semi-supervised learning for these labels.

This script creates 5 directories: 
    rellis-3d-split -> train -> rgb,
    rellis-3d-split -> train -> id, 
    rellis-3d-split -> test -> rgb,
    rellis-3d-split -> test -> id,
    rellis-3d-split -> unlabled
Each newly made directory is checked for duplicates (different file extensions)
and 1 file of the duplicates is kept.

Should probably change this to use glob.glob instead of rglob
so I can avoid using PosixPath objects
"""
import os
import shutil
import numpy as np
from collections import Counter
from pathlib import Path 


percent_split = 0.3 # 30% test split data

# Define root paths to dataset and the paths to the new direcotries
ROOT_DIR = os.getcwd() # path to this script
# HOME = str(Path.home()) # Root of computer
HOME = str('..') # Root of deeplabv3-pytorch repo
RELLIS_PATH = os.path.join(HOME, 'data/rellis/Rellis_3D_pylon_camera_node')
NEW_DATA_ROOT = os.path.join(HOME, 'data/rellis/rellis-3D-camera-split') # dir that will be created
RGB_TRAIN = os.path.join(NEW_DATA_ROOT, 'train/rgb')
ID_TRAIN = os.path.join(NEW_DATA_ROOT,'train/id')
RGB_TEST = os.path.join(NEW_DATA_ROOT,'test/rgb')
ID_TEST = os.path.join(NEW_DATA_ROOT,'test/id')
UNLABELED = os.path.join(NEW_DATA_ROOT,'unlabeled')

VIDEOS = ['00000', '00001', '00002', '00003', '00004']
RGB_IMAGES = os.path.join(RELLIS_PATH, '%s', 'pylon_camera_node')
ID_IMAGES = os.path.join(RELLIS_PATH, '%s', 'pylon_camera_node_label_id')

assert os.path.isdir(RELLIS_PATH)

# Delete dir if exists for a fresh start
shutil.rmtree(NEW_DATA_ROOT, ignore_errors=True)
# Create dir to store train/test split and unlabeled images
print(f'\nCreating 5 new directories at: {NEW_DATA_ROOT}')
Path(RGB_TRAIN).mkdir(parents=True, exist_ok=True) 
Path(RGB_TEST).mkdir(parents=True, exist_ok=True) 
Path(ID_TRAIN).mkdir(parents=True, exist_ok=True)
Path(ID_TEST).mkdir(parents=True, exist_ok=True)
Path(UNLABELED).mkdir(parents=True, exist_ok=True)

def split_train_test_files(image_names, label_names, test_ratio=percent_split):
    """Split image and label file names into train/test sets. 
    Make sure they are sorted first
    """
    shuffled_indices = np.random.permutation(len(label_names))
    test_set_size = int(len(label_names) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    
    image_names = np.array(image_names) # making numpy array to return multiple indices
    label_names = np.array(label_names) # regular python does not support this

    return (image_names[train_indices], image_names[test_indices],
           label_names[train_indices], label_names[test_indices])

def remove_dupe_files(files_list):
    """Remove duplicate files in a list but keep at least 1.
    Example: if there are 3 duplicate files (most likely due to different file extensions),
    delete 2 of these files. Should probably change this to use glob.glob
    so you won't need to pass a posixpath object.

    Args:
        files_stem_list (list): List of file paths which have a .stem property.
    """
    #id_files = list(Path(dir_path).rglob('*.*'))
    files_stem = [x.stem for x in files_list]
    # Create count of each item in list, then remove non-duplicates
    count_dict = dict(Counter(files_stem)) 
    dupe_count = {key:val for key, val in count_dict.items() if val > 1}
    for key, value in dupe_count.items():
        dupe_indices = [i for i, s in enumerate(files_stem) if key in s]
        for dupe in dupe_indices[:-1]:
            del files_list[dupe]
            print('Removing file {0}'.format(files_stem[dupe]))


id_files = []
id_files_stem = []
for video_number in VIDEOS: # Loop through each of the 5 vdieos
    print('Splitting Video: {0}'.format(video_number))
    # Get list of id label images (grayscale ground truth images)
    id_files.append(list(Path(ID_IMAGES % video_number).rglob('*.*')))

id_files = [item for sublist in id_files for item in sublist] # flatten 2d list to 1d, for some reason numpy was not working
id_files = sorted(id_files)
id_files_stem = [x.stem for x in id_files]

rgb_files = [] # list of rgb files full path w/ corresponding label image
rgb_files_stem = []
rgb_unlabeled = [] # list of rgb files full path w/o label image
rgb_unlabeled_stem = []
for video_number in VIDEOS: # Loop through each of the 5 videos
    for file in Path(RGB_IMAGES % video_number).rglob('*.*'):  # Loop through rgb images
        # If the rgb file name is in the id label list - using stem bc images are png and jpg
        # append the image to the rgb train/test list, append to the unlabeled list
        if file.stem in id_files_stem: 
            rgb_files.append(file)
            rgb_files_stem.append(file.stem)
        else:
            rgb_unlabeled.append(file)
            rgb_unlabeled_stem.append(file.stem)

# Sort rgb files to ensure their index aligns with the sorted id labels - this is important for removing duplicates
rgb_files = sorted(rgb_files)
rgb_files_stem = sorted(rgb_files_stem)
# Do not need to sort the unlabeled images because they do not have a matching label image

# Remove/Check for duplicate files in the newly made folders
print('\nChecking for duplicate files due to different file extensions (.png/.jpg)')
remove_dupe_files(id_files)
remove_dupe_files(rgb_files)
remove_dupe_files(rgb_unlabeled)

print(f'\nSplitting RGB and Id images to a {(1-percent_split)*100}% train and {percent_split*100}% test set')
X_train, X_test, y_train, y_test = split_train_test_files(rgb_files, id_files, percent_split)


print('\nCopying train/test split files to their new directory')
for file in X_train:
    shutil.copyfile(file, os.path.join(RGB_TRAIN, file.name))
for file in X_test:
    shutil.copyfile(file, os.path.join(RGB_TEST, file.name))
for file in y_train:
    shutil.copyfile(file, os.path.join(ID_TRAIN, file.name))
for file in y_test:
    shutil.copyfile(file, os.path.join(ID_TEST, file.name))
for file in rgb_unlabeled:
    shutil.copyfile(file, os.path.join(UNLABELED, file.name))
