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
    def __init__(self, args):
        super().__init__()

        datadir = args.dataset_path
        transforms = args.transforms

        self.imgdir = os.path.join(datadir, "images") #Change to images/rgb/ later
        self.imgpaths = sorted(os.listdir(self.imgdir))
        self.img_metadata = json.load(open(os.path.join(datadir, "ind_map.json"))) #Gives image height/width and other useful info

        self.labeldir = os.path.join(datadir, "labels")
        self.labelpaths = sorted(os.listdir(self.labeldir))

        if args.use_line_seg: #If we want to concatenate line segmentations to input
            self.line_seg_dir = os.path.join(datadir, "images", "line_seg")
            self.line_seg_paths = sorted(os.listdir(self.line_seg_dir))
        
        if args.use_seg_mask: #If we want to concatenate segmentation masks to input
            self.seg_mask_dir = os.path.join(datadir, "images", "mask")
            self.seg_mask_paths = sorted(os.listdir(self.seg_mask_dir))


        self.gt_voxels = read_hdf5(os.path.join(datadir, "off_models_32_x_32", "output.h5"))

        self.imgs = []
        self.imgnums = []

        for imgpath, labelpath in zip(self.imgpaths, self.labelpaths):
            img = cv2.imread((os.path.join(self.imgdir, imgpath)))

            if args.use_line_seg:
                line_seg_img = cv2.imread(os.path.join(self.line_seg_dir, imgpath))
                img = np.dstack([img, line_seg_img]) #Stack RGB img and line segmentation depth-wise
            if args.use_seg_mask:
                seg_mask_img = cv2.imread(os.path.join(self.seg_mask_dir, imgpath))
                img = np.dstack([img, seg_mask_img]) #Stack RGB img and segmentation mask depth-wise

            img_data = self.img_metadata[imgpath]
            h, w = img_data["img_h"], img_data["img_w"]

            with open(os.path.join(self.labeldir, labelpath)) as f: #Read in label txt file for specific image
                for line in f.readlines(): #For each bbox in image
                    line = np.array(line.split(" ")).astype(np.float32).reshape(-1) # [class, x_c, y_c, w, h]
                    line = torch.from_numpy(line)
                    if line[0] == 1: #If bounded region is relevant
                        bbox_x_center = line[1]
                        bbox_y_center = line[2]
                        bbox_w = line[3]
                        bbox_h = line[4]

                        #Convert x_c, y_c, w, h to start_x, start_y, end_x, end_y 
                        start_x = torch.floor(w*(bbox_x_center - bbox_w/2)).to(int).item()
                        start_y = torch.floor(h*(bbox_y_center - bbox_h/2)).to(int).item()

                        end_x = torch.floor(w*(bbox_x_center + bbox_w/2)).to(int).item()
                        end_y = torch.floor(h*(bbox_y_center + bbox_h/2)).to(int).item()

                        #Get bounded region
                        cropped_img = img[start_y: end_y, start_x: end_x]
                        
                        #Convert to PIL image for PyTorch transforms
                        cropped_img = Image.fromarray(cropped_img)

                        #Apply transforms if they exist
                        if transforms is not None:
                            cropped_img = transforms(cropped_img)

                        self.imgnums.append(int(imgpath.split(".")[0]))
                        self.imgs.append(cropped_img)
    
        assert len(self.imgnums) == len(self.imgs)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        dic = {
            'image_num': self.imgnums[idx],
            'image': self.imgs[idx],
            'voxels': self.gt_voxels[idx],
        }
        
        return dic
