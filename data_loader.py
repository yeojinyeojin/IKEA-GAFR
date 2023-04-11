import os
import sys
import argparse
from glob import glob
import json

import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision import models as torchvision_models
import pytorch3d
from pytorch3d.io import load_obj

class IKEAManualPDF(Dataset):
    def __init__(self, datadir, transforms=None):
        super().__init__()
        
        self.imgdir = f"{datadir}/images"
        self.imgpaths = glob(f"{self.imgdir}/**/*.png", recursive=True)
        
        
        self.annotfile = f"{datadir}/annotation.json" #TODO
        with open(self.annotfile) as f:
            self.annotation = json.load(f)
        
        self.imgs = []
        for imgpath in self.imgpaths:
            img = Image.open(imgpath)
            if transforms is not None:
                img = transforms(img)
            self.imgs.append(img)
        # sorted(self.imgs) #TODO: needed?
        
        #TODO: load labels
        self.labels = torch.randn(len(self.imgs), 5)
        # self.labels = []
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        dic = {
            'images': self.imgs[idx],
            'voxels': self.voxels[idx],
        }
        
        return dic

class IKEAManualStep(Dataset):
    def __init__(self, datadir, transforms=None):
        super().__init__()
        
        self.imgdir = f"{datadir}/images"
        #self.imgpaths = glob(f"{self.imgdir}/**/*.png", recursive=True)
        self.imgpaths = sorted(os.listdir(self.imgdir))

        self.modeldir = f"{datadir}/models"
        #self.modelpaths = glob(f"selg.modeldir}/**/*.obj", recursive=True)
        self.modelpaths = sorted(os.listdir(self.modeldir))

        self.imgs = []
        for imgpath in self.imgpaths:
            img = Image.open(os.path.join(self.imgdir, imgpath))
            if transforms is not None:
                img = transforms(img)
            self.imgs.append(img)
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        dic = {
            'images': self.imgs[idx],
            'voxels': self.voxels[idx],
        }
        
        return dic
