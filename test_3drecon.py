import os
import sys
import time
import argparse
from tqdm import tqdm
from datetime import datetime
import h5py
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision import models as torchvision_models
import pytorch3d
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
from pytorch3d.io import load_obj, save_obj
from pytorch3d.ops import cubify

from model import Pix2Voxel
from train_utils import create_dir
from data_loader import IKEAManualStep, R2N2
from train_3drecon import preprocess, calculate_voxel_loss

def parse_args():
    
    parser = argparse.ArgumentParser('IKEA23D_inference', add_help=False)
    
    # Model parameters
    parser.add_argument('--encoder_arch', default='resnet18', type=str, help="architecture for encoder")
    parser.add_argument('--model', default='pix2vox', type=str, help="architecture for 3D reconstruction")
    
    # Pre-Training parameters
    parser.add_argument('--r2n2', default=False, action='store_true')
    parser.add_argument('--r2n2_dir', default='./dataset/r2n2_shapenet_dataset', type=str)
    
    # Training parameters
    parser.add_argument('--max_iter', default=10000, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--lr', default=4e-4, type=str)
    parser.add_argument('--resize_w', default=224, type=int)
    parser.add_argument('--resize_h', default=224, type=int)
    parser.add_argument('--use_line_seg', action=argparse.BooleanOptionalAction)
    parser.add_argument('--use_seg_mask', action=argparse.BooleanOptionalAction)
    
    # Logging parameters
    parser.add_argument('--log_freq', default=1, type=str)
    
    parser.add_argument('--device', default='cuda', type=str) 
    
    # Directories & Checkpoint
    parser.add_argument('--dataset_path', type=str, default='./dataset')
    parser.add_argument('--load_checkpoint', default='./checkpoints/pix2vox/checkpoint_3000.pth', type=str)            
    parser.add_argument('--out_dir', type=str, default='./inference_outputs')
    
    return parser.parse_args()

def main(args):
    
    create_dir(args.out_dir)
    
    ## Load model & optimizer
    if args.model == 'pix2vox':
        model = Pix2Voxel(args).to(args.device)
    model.eval()
    
    checkpoint = torch.load(args.load_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"@@@@ Succesfully loaded model")
    
    ## Load dataset
    transform = transforms.Compose([
        transforms.Resize(tuple((args.resize_w, args.resize_h))),
        transforms.ToTensor()
    ]) #TODO: normalize?
    args.transforms = transform
    
    if args.r2n2:
        shapenet_path = f"{args.r2n2_dir}/shapenet"
        r2n2_path = f"{args.r2n2_dir}/r2n2"
        splits_path = f"{args.r2n2_dir}/line_split.json"
        
        dataset = R2N2("test", 
                        shapenet_path, r2n2_path, splits_path, 
                        return_voxels=True, return_feats=False, return_RTK=False,
                        views_rel_path="LineDrawings")
        
        test_dataloader = DataLoader(dataset,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    collate_fn=collate_batched_R2N2,
                                    pin_memory=True,
                                    drop_last=True,
                                    )
    
    else:
        dataset = IKEAManualStep(args)
        
        test_dataloader = DataLoader(dataset=dataset, 
                                    batch_size=args.batch_size, 
                                    shuffle=True, 
                                    num_workers=args.num_workers)

    print (f"@@@@ Successfully loaded data {len(dataset)}")
    
    num_batches, total_loss, predictions, image_names = 0, 0, None, None
    start_time = time.time()
    for i, batch in enumerate(test_dataloader):
        step_start_time = time.time()
        
        read_start_time = time.time()
        image_test_names, images_gt, ground_truth_3d = preprocess(batch, args)
        read_time = time.time() - read_start_time
        
        with torch.no_grad():
            prediction_3d = model(images_gt, args).squeeze()
        # if predictions is None:
        #     predictions = prediction_3d
        #     image_names = image_test_names
        # else:
        #     predictions = torch.cat([predictions, prediction_3d], dim=0)
        #     image_names = torch.cat([image_names, image_test_names], dim=0)
        prediction_3d = prediction_3d.unsqueeze(0)
        loss = calculate_voxel_loss(prediction_3d, ground_truth_3d)
        total_loss += loss
        loss_vis = loss.cpu().item()
        
        total_time = time.time() - start_time
        iter_time = time.time() - step_start_time
        
        num_batches += 1
        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f" % (i, args.max_iter, total_time, read_time, iter_time, loss_vis))
        
        # Volume Visualization
        mesh = cubify(prediction_3d, thresh=0)
        save_obj(f"{args.out_dir}/{i}.obj", verts=mesh.verts_list()[0], faces=mesh.faces_list()[0])
        import pdb; pdb.set_trace()
        # tex = torch.ones_like(mesh.verts_list()[0]) * torch.tensor([0,0,0]).to(args.device)
        # mesh.textures = pytorch3d.renderer.TexturesVertex(tex.unsqueeze(0))
        # rend = renderer(entity, cameras=cameras.to(device), lights=lights.to(device))

    if num_batches != 0:
        total_loss /= num_batches
    print("Total Loss: ", total_loss)
        
        
if __name__ == "__main__":
    args = parse_args()
    main(args)