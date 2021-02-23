"""
# Copyright (c) Microsoft Corporation.
# Copyright (c) Jedrzej Kozerawski
# Licensed under the MIT license.

Partially based on https://github.com/zhmiao/OpenLongTailRecognition-OLTR
@author: Zhongqi Miao
Copyright (c) 2019, Zhongqi Miao
# Licensed under the BSD 3-Clause License.
"""

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import torch
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Data transformation with augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Dataset
class LT_Dataset(Dataset):
    
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                if 'iNaturalist' in root:
                    img_path = str(line.split()[0])[2:-1]
                else:
                    img_path = line.split()[0]
                self.img_path.append(os.path.join(root, img_path))
                self.labels.append(int(line.split()[1]))
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label, path

# Load datasets
def load_data(data_root, dataset, phase, batch_size, use_sampler = False, num_workers=4, shuffle=True, gamma=1.0):
    txt = './data/%s/%s_%s.txt'%(dataset, dataset, phase)
    print('Loading data from %s' % (txt))
    transform = data_transforms[phase]
    print('Use data transformation:', transform)

    set_ = LT_Dataset(data_root, txt, transform)

    if phase == 'train' and use_sampler:

        # Prepare the sampler:
        print('Using sampler.')
        txt_file = txt
        with open(txt_file, 'r') as txtfile:
            lines = txtfile.readlines()
        num_classes = -1
        for line in lines:
            lbl = int(line.split()[-1])
            if lbl > num_classes:
                num_classes = lbl
        label_count = np.zeros(num_classes+1)
        for line in lines:
            lbl = int(line.split()[-1])
            label_count[lbl] += 1
        all_images = np.sum(label_count)
        weights = []
        for line in lines:
            lbl = int(line.split()[-1])
            weights.append(all_images / (label_count[lbl]**gamma))
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        return DataLoader(dataset=set_, batch_size=batch_size, shuffle=False, sampler=sampler, num_workers=num_workers, pin_memory=True)
    else:
        print('No sampler.')
        print('Shuffle is %s.' % (shuffle))
        return DataLoader(dataset=set_, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
        
    
    
