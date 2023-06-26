import os
import torch
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader

class LocalizationDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.items = sorted(os.listdir(self.root))

    def __len__(self):
        return len(os.listdir(self.root))

    def __repr__(self):
        return f"Localization and Denoising Dataset: {self.__len__()} PA Frames"
    
    def __getitem__(self, index):
        img = np.array(loadmat(os.path.join(self.root, self.items[index]))['PA_Image'])
        return torch.Tensor(img)

def test():
    dataset = LocalizationDataset(root = './data/')
    print(dataset, dataset.items, len(dataset))

if __name__ == "__main__":
    test()