import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from models import *

if __name__ == "__main__":
    np.random.seed(9)
    torch.manual_seed(9)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    device = "cuda" if torch.cuda.is_available() else "cpu"