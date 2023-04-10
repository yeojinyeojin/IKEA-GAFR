import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models as torchvision_models
import pytorch3d

class RelevanceClassifier(nn.Module):
    def __init__(self, args):
        super(RelevanceClassifier, self).__init__()
        
    def forward(self, x):
        pass

class Pix2Voxel(nn.Module):
    def __init__(self, args):
        super(Pix2Voxel, self).__init__()
        
        self.device = args.device
        
        if not args.load_feat: #pretrained ResNet18 features of images
            vision_model = torchvision_models.__dict__[args.encoder_arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(16, 8, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(8, 4, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(4),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(4, 1, kernel_size=1),
            torch.nn.BatchNorm3d(1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, images, args):
        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size
        
        inp = encoded_feat.view(-1, 64, 2, 2, 2)
        voxels_pred = self.decoder(inp)
        return voxels_pred