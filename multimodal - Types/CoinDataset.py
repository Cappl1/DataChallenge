# Reference: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
import os
import torch
import pandas as pd
from PIL import Image
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F




class CoinDataset(Dataset):
    """GeoGuessr dataset."""

    def __init__(self, csv_file, embedding_path, transform=None, num_classes=121, ):
        """ 
         Args:
            - csv_file (string): Path to the csv file with image paths and associated labels.
            - embedding_path (string): Path to the numpy file with precomputed text embeddings.
            - transform (callable, optional): Optional transform to be applied on an image.
                This could include operations such as normalization, data augmentation etc.
            - num_classes (int, optional): Number of different classes or types of coins.
                This parameter is used for the one-hot encoding of labels. Defaults to 121.
        """
        self.num_classes = num_classes
        self.data = pd.read_csv(csv_file)
        self.embedding = np.load(embedding_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        
        image = Image.open(os.path.join(self.data.iloc[idx, 0],self.data.iloc[idx, 1]))
        label = self.data.iloc[idx, 3]
        text_embedding = torch.tensor(self.embedding[label , np.random.randint(0,10)])
        
        
        # Convert to one-hot vector
        label = F.one_hot(torch.tensor(label), num_classes=self.num_classes)
        
        sample = {'image': image, 'label': label, 'text_embedding': text_embedding}

        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample
    
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        
        # not 100% sure if transforming y is of any use yet
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(np.array(label))}
    
