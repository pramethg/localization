import os
import torch
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset

class LocalizationDataset(Dataset):
    def __init__(self, root = './data/', transforms = None):
        self.root = root
        self.transforms = transforms
        self.items = sorted(os.listdir(os.path.join(self.root, "images")))
        self.labels = sorted(os.listdir(os.path.join(self.root, "labels")))

    def __len__(self):
        return len(self.items)

    def __repr__(self):
        return f"Localization and Denoising Dataset: {self.__len__()} PA Frames"

    @staticmethod
    def normalize(image):
        mean = np.mean(image, axis = (0, 1))
        std = np.std(image, axis = (0, 1))
        normalized_image = (image - mean) / std
        scaled_image = (normalized_image - np.min(normalized_image)) / (np.max(normalized_image) - np.min(normalized_image))
        return scaled_image

    def __getitem__(self, index):
        img = np.array(loadmat(os.path.join(self.root, self.items[index]))['PA_Image'])
        label = np.array(loadmat(os.path.join(self.root, self.labels[index]))['PA_Label'])
        img = self.normalize(img)
        img = self.transforms(img) if self.transforms is not None else img
        return img, torch.Tensor(label)

def test():
    dataset = LocalizationDataset(root = './data/')
    print(dataset, dataset.items, len(dataset))

if __name__ == "__main__":
    test()
