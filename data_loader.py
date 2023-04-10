import os
import sys
import argparse
from glob import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision import models as torchvision_models
import pytorch3d

class IKEAManual(Dataset):
    def __init__(self, datadir):
        super().__init__()
        
        self.imgdir = f"{datadir}/images"
        self.imgpaths = glob(f"{self.imgdir}/**/*.png", recursive=True)
        
        self.imgs = []
        for imgpath in self.imgpaths:
            img = cv2.imread(imgpath)
            self.imgs.append(img)
        # sorted(self.imgs) #TODO: needed?
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx]