# Dataset and Dataloader [train, val, test]
# CNN model with hyperparameter optimization
# 

import os
import torch
import pickle
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from config import BASE_DIR, N_ATTRIBUTES
from torch.utils.data import BatchSampler
from torch.utils.data import Dataset, DataLoader
import csv

IMG_DIRECTORY_CUB_200_2011 ="/home/anjilabudathoki/dip-project/project-2025/src/"


class CUBDataset(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CUB dataset
    """

    def __init__(self, pkl_file_paths, use_attr, no_img, uncertain_label, image_dir, n_class_attr, transform=None):
        """
        Arguments:
        pkl_file_paths: list of full path to all the pkl data
        use_attr: whether to load the attributes (e.g. False for simple finetune)
        no_img: whether to load the images (e.g. False for A -> Y model)
        uncertain_label: if True, use 'uncertain_attribute_label' field (i.e. label weighted by uncertainty score, e.g. 1 & 3(probably) -> 0.75)
        image_dir: default = 'images'. Will be append to the parent dir
        n_class_attr: number of classes to predict for each attribute. If 3, then make a separate class for not visible
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """
        self.data = []
        self.is_train = any(["train" in path for path in pkl_file_paths])
        if not self.is_train:
            assert any([("test" in path) or ("val" in path) for path in pkl_file_paths])
        for file_path in pkl_file_paths:
            self.data.extend(pickle.load(open(file_path, 'rb')))
        self.transform = transform
        self.use_attr = use_attr
        self.no_img = no_img
        self.uncertain_label = uncertain_label
        self.image_dir = image_dir
        self.n_class_attr = n_class_attr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        
        img_path = img_data['img_path']
        # try:
        idx = img_path.split('/').index('CUB_200_2011')
        cub_200_2011_path_arr = img_path.split('/')[idx:]
        cub_200_2011_path_arr.insert(0, 'datasets')
        img_path = '/'.join(cub_200_2011_path_arr)
        img_path  = IMG_DIRECTORY_CUB_200_2011 + img_path
        img = Image.open(img_path).convert('RGB')
            
        # except:
           
            # img_path_split = img_path.split('/')
            # split = 'train' if self.is_train else 'test'
            # img_path = '/'.join(img_path_split[:2] + [split] + img_path_split[2:])
            # img = Image.open(img_path).convert('RGB')
        class_label = img_data['class_label']
        
        if self.transform:
            img = self.transform(img)

        if self.use_attr:
            if self.uncertain_label:
                attr_label = img_data['uncertain_attribute_label']
            else:
                attr_label = img_data['attribute_label']
            if self.no_img:
                if self.n_class_attr == 3:
                    one_hot_attr_label = np.zeros((N_ATTRIBUTES, self.n_class_attr))
                    one_hot_attr_label[np.arange(N_ATTRIBUTES), attr_label] = 1
                    return one_hot_attr_label, class_label
                else:
                    return attr_label, class_label
            else:
                return img, class_label, attr_label
        else:
            return img, class_label



def get_data(pkl_paths, use_attr, no_img, batch_size, uncertain_label=False, n_class_attr=2, image_dir='images', resampling=False, resol=299, augmentation_type='standard'):
    resized_resol = int(resol * 256/224) # we have used inception so resizing image ? 
    is_training = any(['train.pkl' in f for f in pkl_paths]) # # choosing training dataset
    
    # While training, we use different image processing.
    if is_training: 
        if augmentation_type == 'standard':
            transform = transforms.Compose([
                transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
                transforms.RandomResizedCrop(resol),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*3, std=[2]*3)
            ])
        elif augmentation_type == 'grayscale':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(resol),
                transforms.RandomHorizontalFlip(),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*3, std=[2]*3)
            ])
        elif augmentation_type == 'rotation':
            transform = transforms.Compose([
                transforms.RandomRotation(degrees=30),
                transforms.RandomResizedCrop(resol),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*3, std=[2]*3)
            ])
        elif augmentation_type == 'center_crop':
            transform = transforms.Compose([
                transforms.Resize((resized_resol, resized_resol)),
                transforms.CenterCrop(resol),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*3, std=[2]*3)
            ])
        elif augmentation_type == 'custom':
            # Define your own custom sequence of transforms
            transform = transforms.Compose([
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                transforms.ColorJitter(contrast=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(resol),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*3, std=[2]*3)
            ])
        else:
            raise ValueError(f"Unsupported augmentation type: {augmentation_type}")
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(resol),
            transforms.ToTensor(), #implicitly divides by 255
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
            ])
    dataset = CUBDataset(pkl_paths, use_attr, no_img, uncertain_label, image_dir, n_class_attr, transform)
    
    if is_training:
        drop_last = True
        shuffle = True
    else:
        drop_last = False
        shuffle = False
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return loader

DATA_DIR = '/home/anjilabudathoki/dip-project/project-2025/src/datasets/CUB_processed/class_attr_data_10'
train_data_path = os.path.join(BASE_DIR, DATA_DIR, 'train.pkl')
val_data_path = train_data_path.replace('train.pkl', 'val.pkl')

# args.use_attr = False
# args.no_img  = False
# args.batch_size = 16
# args.uncertain_labels = 
# args.image_dir

train_loader = get_data([train_data_path], False, True, 16, False, image_dir='image', \
                                 n_class_attr=112, resampling=False)
val_loader = get_data([val_data_path], False, True, 16, image_dir='images', n_class_attr=112)

# for epoch in range(0,5):  
    