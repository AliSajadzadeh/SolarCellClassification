from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import os


train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self, data,mode):
        self.data = data
        self.mode = mode
        if not isinstance(self.mode,str):
            raise TypeError('mode should be a string')
        self._transform = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        labels = self.data.loc[index,['crack','inactive']]
        image_labels = self.data.loc[index,['filename']]
        image_labels = image_labels.str.split('/',expand = True)[1]
        path = 'images/'
        path = path + image_labels
        for f in path:
            images = imread(f)
        images = gray2rgb(images)
        images = images.swapaxes(1,2)
        images = images.swapaxes(0,1)
        images = torch.tensor(images)
        #labels_output = torch.tensor(labels)
        if self.mode == 'train':
            self._transform = tv.transforms.Compose([tv.transforms.Resize(300), tv.transforms.RandomRotation(90)
                                                     , tv.transforms.RandomHorizontalFlip,tv.transforms.RandomVerticalFlip,
                                                     tv.transforms.ToPILImage(),tv.transforms.ToTensor(),
                                                     tv.transforms.Normalize(train_mean,train_std)])
        else:
            self._transform = tv.transforms.Compose([tv.transforms.ToPILImage(), tv.transforms.ToTensor(),
                                                     tv.transforms.Normalize(train_mean, train_std)])

        images_output = self._transform(images)
        return images_output, torch.tensor(labels)




