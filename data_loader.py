import os
import sys
import argparse
from glob import glob
import json
import h5py
import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision import models as torchvision_models
# import pytorch3d
# from pytorch3d.io import load_obj
def read_hdf5(file, key = 'tensor'):
    assert os.path.exists(file), 'file %s not found' % file
    h5f = h5py.File(file, 'r')
    assert key in h5f.keys(), 'key %s not found in file %s' % (key, file)
    gt_voxels = torch.from_numpy(h5f[key][()])
    h5f.close()

    return gt_voxels

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
        
        self.imgdir = os.path.join(datadir, "images")
        self.imgpaths = sorted(os.listdir(self.imgdir))
        self.img_metadata = json.load(open(os.path.join(datadir, "ind_map.json")))

        self.labeldir = os.path.join(datadir, "labels")
        self.labelpaths = sorted(os.listdir(self.labeldir))

        self.gt_voxels = read_hdf5(os.path.join(datadir, "off_models_32_x_32", "output.h5"))
        #self.modelpaths = glob(f"selg.modeldir}/**/*.obj", recursive=True)
        #self.modelpaths = sorted(os.listdir(self.gt_voxels))

        self.imgs = []
        counter = 0
        for imgpath, labelpath in zip(self.imgpaths, self.labelpaths):
            img = cv2.imread((os.path.join(self.imgdir, imgpath)))
            img_data = self.img_metadata[imgpath]
            h, w = img_data["img_h"], img_data["img_w"]

            with open(os.path.join(self.labeldir, labelpath)) as f:
                for line in f.readlines():
                    line = np.array(line.split(" ")).astype(np.float32).reshape(-1)
                    line = torch.from_numpy(line)
                    if line[0] == 1:
                        bbox_x_center = line[1]
                        bbox_y_center = line[2]
                        bbox_w = line[3]
                        bbox_h = line[4]

                        start_x = torch.floor(w*(bbox_x_center - bbox_w/2)).to(int).item()
                        start_y = torch.floor(h*(bbox_y_center - bbox_h/2)).to(int).item()

                        end_x = torch.floor(w*(bbox_x_center + bbox_w/2)).to(int).item()
                        end_y = torch.floor(h*(bbox_y_center + bbox_h/2)).to(int).item()
                        #print("Y: ", "Image Height: ", h, "Start: ", start_y, "End: ", end_y)
                        #print("X: ", "Image Width: ", w, "Start: ", start_x, "End; ", end_x)
                        #print()

                        img_curr = img[start_y: end_y, start_x: end_x]
                        
                        img_curr = Image.fromarray(img_curr)
                        if transforms is not None:
                            img_curr = transforms(img_curr)

                        self.imgs.append(img_curr)
                        counter += 1

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        dic = {
            'image': self.imgs[idx],
            'voxels': self.gt_voxels[idx],
        }
        
        return dic
