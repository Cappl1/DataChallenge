# Reference: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F



class GeoGuessrDataset(Dataset):
    """GeoGuessr dataset."""

    def __init__(self, csv_file, root_dir, transform=None, num_classes=32768):
        """
        Args:
            csv_file (string): Path to the csv file with coordinates.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.num_classes = num_classes
        self.coordinates = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.coordinates.iloc[idx, 0])
        image = io.imread(img_name)
        
        #adjusted vor Haversine
        lat = self.coordinates.iloc[idx, 1]
        lon = self.coordinates.iloc[idx, 2]
        ######################
        
        
        geohash = self.coordinates.iloc[idx, 4]


        
        # Convert to one-hot vector
        geohash = F.one_hot(torch.tensor(geohash), num_classes=self.num_classes)
        
        
        #adjusted vor Haversine
        sample = {'image': image, 'geohash': geohash, 'gt': (lat,lon)}
        ##############
        
        
        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample
    
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, geohash = sample['image'], sample['geohash']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        
        # not 100% sure if transforming y is of any use yet
        return {'image': torch.from_numpy(image),
                'geohash': torch.from_numpy(np.array(geohash))}
    
