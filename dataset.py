import os
import torch
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader

class LocalizationDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.items = sorted(os.listdir(os.path.join(self.root, "images")))
        self.labels = sorted(os.listdir(os.path.join(self.root, "labels")))

    def __len__(self):
        return len(self.items)

    def __repr__(self):
        return f"Localization and Denoising Dataset: {self.__len__()} PA Frames"
    
    def __getitem__(self, index):
        img = np.array(loadmat(os.path.join(self.root, self.items[index]))['PA_Image'])
        label = np.array(loadmat(os.path.join(self.root, self.labels[index]))['PA_Label'])
        return torch.Tensor(img), torch.Tensor(label)

def test():
    dataset = LocalizationDataset(root = './data/')
    print(dataset, dataset.items, len(dataset))

if __name__ == "__main__":
    test()