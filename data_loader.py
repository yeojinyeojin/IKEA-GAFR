import os
import sys
import argparse
from glob import glob

import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision import models as torchvision_models
import pytorch3d

class IKEAManual(Dataset):
    def __init__(self, datadir, load_feat=False, transforms=None):
        super().__init__()
        
        self.imgdir = f"{datadir}/images"
        self.imgpaths = glob(f"{self.imgdir}/**/*.png", recursive=True)
        
        self.imgs = []
        for imgpath in self.imgpaths:
            img = Image.open(imgpath)
            if transforms is not None:
                img = transforms(img)
            self.imgs.append(img)
        # sorted(self.imgs) #TODO: needed?
        
        #TODO: load voxels
        self.voxels = self.imgs
        # self.voxels = []
        
        self.load_feat = load_feat
        if load_feat:
            self.feats = torch.from_numpy(np.load(f"{datadir}/feats.npy"))
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        dic = {
            'images': self.imgs[idx],
            'voxels': self.voxels[idx],
        }
        
        if self.load_feat:
            dic['feats'] = self.feats[idx]
        
        return dic
        