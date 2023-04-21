import os
import sys
import time
import argparse
from tqdm import tqdm
from datetime import datetime
import h5py
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision import models as torchvision_models
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
from pytorch3d.io import load_obj, save_obj
from pytorch3d.ops import cubify
from pytorch3d.datasets.r2n2.utils import (
    BlenderCamera,
    align_bbox,
    compute_extrinsic_matrix,
    read_binvox_coords,
    voxelize,
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex, look_at_view_transform
from pytorch3d.vis.plotly_vis import plot_scene

from model import Pix2Voxel
from train_utils import create_dir
from data_loader import IKEAManualStep, R2N2
import utils_vox

def parse_args():
    
    parser = argparse.ArgumentParser('IKEA23D', add_help=False)
    
    # Model parameters
    parser.add_argument('--encoder_arch', default='resnet18', type=str, help="architecture for encoder")
    parser.add_argument('--model', default='pix2vox', type=str, help="architecture for 3D reconstruction")
    
    # Pre-Training parameters
    parser.add_argument('--r2n2', default=True, action='store_true')
    parser.add_argument('--r2n2_dir', default='./dataset/r2n2_shapenet_dataset', type=str)
    
    # Training parameters
    parser.add_argument('--max_iter', default=10000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--lr', default=4e-4, type=str)
    parser.add_argument('--train_test_split_ratio', default=0.7, type=float)
    parser.add_argument('--resize_w', default=224, type=int)
    parser.add_argument('--resize_h', default=224, type=int)
    parser.add_argument('--use_line_seg', action=argparse.BooleanOptionalAction)
    parser.add_argument('--use_seg_mask', action=argparse.BooleanOptionalAction)
    parser.add_argument('--debug', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--chairs_only', default=False, action=argparse.BooleanOptionalAction)
    
    # Logging parameters
    parser.add_argument('--log_freq', default=1000, type=str)
    parser.add_argument('--save_freq', default=500, type=int)    
    parser.add_argument('--eval_freq', default=1000, type=int)
    
    parser.add_argument('--device', default='cuda', type=str) 
    
    # Directories & Checkpoint
    parser.add_argument('--load_checkpoint', default=None, type=str)            
    # parser.add_argument('--load_checkpoint', default='./checkpoints/pix2vox/r2n2_pretraining/checkpoint_2500.pth', type=str)            
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--logs_dir', type=str, default='./logs')
    parser.add_argument('--debug_dir', type=str, default='./debug_output')
    parser.add_argument('--out_dir', type=str, default='./train_output')
    parser.add_argument('--dataset_path', type=str, default='./dataset')
    
    return parser.parse_args()

def preprocess(feed_dict, args, dataset):
    image_names = torch.tensor(feed_dict['names']).reshape(-1, 1)
    images = feed_dict['images'].squeeze(1).float()
    voxels = feed_dict['voxels'].float()
    ground_truth_3d = voxels
    
    return image_names.to(args.device), images.to(args.device), ground_truth_3d.to(args.device)

def calculate_voxel_loss(voxel_src, voxel_tgt):
    voxel_src = torch.clip(voxel_src, min=0, max=1)
    voxel_tgt = torch.clip(voxel_tgt, min=0, max=1)
    loss = torch.nn.BCELoss()(voxel_src, voxel_tgt)
    
    return loss

def save_as_mesh(prediction_3d, save_name, cubify_thresh=0.5):
    mesh = cubify(prediction_3d, thresh=cubify_thresh)
    save_obj(f"{save_name}.obj", verts=mesh.verts_list()[0], faces=mesh.faces_list()[0])
    
def main(args):
    ## Create directories for checkpoints and tensorboard logs
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.model)
    args.logs_dir = os.path.join(args.logs_dir, args.model)
    create_dir(args.checkpoint_dir)
    create_dir(args.logs_dir)
    
    ## Tensorboard Logger
    dt = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    create_dir(os.path.join(args.logs_dir, dt))
    args.out_dir = os.path.join(args.out_dir, dt)
    create_dir(args.out_dir)
    args.debug_dir = os.path.join(args.debug_dir, dt)
    create_dir(args.debug_dir)
    writer = SummaryWriter(os.path.join(args.logs_dir, dt))
    
    ## Load model & optimizer
    if args.model == 'pix2vox':
        model = Pix2Voxel(args).to(args.device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    
    if args.load_checkpoint is not None:
        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint['step']
        print(f"@@@@ Succesfully loaded iter {start_iter}")
    else:
        start_iter = 0
        
    ## Load dataset
    transform = transforms.Compose([
        transforms.Resize(tuple((args.resize_w, args.resize_h))),
        transforms.ToTensor()
    ]) #TODO: normalize?
    args.transforms = transform

    if args.r2n2:
        # shapenet_path = f"{args.r2n2_dir}/shapenet"
        # r2n2_path = f"{args.r2n2_dir}/r2n2"
        # splits_path = f"{args.r2n2_dir}/line_split.json"
        
        # train_set = R2N2(
        #                 # "train", 
        #                 shapenet_path, r2n2_path, 
        #                 # splits_path, 
        #                 # return_voxels=True, return_feats=False, return_RTK=False,
        #                 views_rel_path="LineDrawings", voxels_rel_path="ShapeNetVoxels")
        # test_set = R2N2(
        #                 # "test", 
        #                 shapenet_path, r2n2_path, 
        #                 # splits_path, 
        #                 # return_voxels=True, return_feats=False, return_RTK=False,
        #                 views_rel_path="LineDrawings", voxels_rel_path="ShapeNetVoxels")

        # train_dataloader = DataLoader(
        #     train_set,
        #     batch_size=args.batch_size,
        #     num_workers=args.num_workers,
        #     collate_fn=collate_batched_R2N2,
        #     pin_memory=True,
        #     drop_last=True,
        #     )
        # test_dataloader = DataLoader(
        #     test_set,
        #     batch_size=1,
        #     num_workers=args.num_workers,
        #     collate_fn=collate_batched_R2N2,
        #     pin_memory=True,
        #     drop_last=True,
        #     )
        # train_loader = iter(train_dataloader)
        # test_loader = test_dataloader

        shapenet_path = f"{args.r2n2_dir}/shapenet"
        r2n2_path = f"{args.r2n2_dir}/r2n2"
        metadata_path = f"{args.r2n2_dir}/line_metadata.json"
        
        dataset = R2N2(shapenet_path, r2n2_path, metadata_path,
                       views_rel_path="LineDrawings", voxels_rel_path="ShapeNetVoxels")
    else:
        dataset = IKEAManualStep(args, chairs_only=args.chairs_only)

    train_set, test_set = random_split(dataset, [args.train_test_split_ratio, 1-args.train_test_split_ratio])

    train_dataloader = DataLoader(dataset=train_set, 
                            batch_size=args.batch_size, 
                            shuffle=True, 
                            num_workers=args.num_workers)

    test_dataloader = DataLoader(dataset=test_set, 
                            batch_size=args.batch_size, 
                            shuffle=True, 
                            num_workers=args.num_workers)

    train_loader = iter(train_dataloader)
    test_loader = test_dataloader
    
    print (f"@@@@ Successfully loaded data: train_set {len(train_set)} | test_set {len(test_set)}")
    
    print("@@@@ Starting training!")
    
    start_time = time.time()
    for step in range(start_iter, args.max_iter):
        step_start_time = time.time()

        if step % len(train_loader) == 0: #restart after one step
            train_loader = iter(train_dataloader)

        read_start_time = time.time()

        feed_dict = next(train_loader)

        names, images_gt, ground_truth_3d = preprocess(feed_dict, args, dataset)
        read_time = time.time() - read_start_time

        prediction_3d = model(images_gt, args).squeeze()
        
        if args.debug and step % len(train_loader) == 0:
            for i, name in enumerate(names):
                save_as_mesh(prediction_3d[i].unsqueeze(0), f"{args.debug_dir}/{step}_{name.item()}")
                save_as_mesh(ground_truth_3d[i].unsqueeze(0), f"{args.debug_dir}/{step}_{name.item()}_gt")
                cv2.imwrite(f"{args.debug_dir}/{step}_{name.item()}_gt.png", images_gt[i].permute(0,2,3,1).cpu().numpy())
                break

        loss = calculate_voxel_loss(prediction_3d, ground_truth_3d)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - step_start_time

        loss_vis = loss.cpu().item()
        
        writer.add_scalar('train_loss', loss_vis, step)

        if ((step + 1) % args.save_freq) == 0:
            print("saved at ", step)
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(args.checkpoint_dir, 'checkpoint_{}.pth'.format(step+1)))

        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f" % (step, args.max_iter, total_time, read_time, iter_time, loss_vis))

        if ((step + 1) % args.eval_freq) == 0:
            print("@@@ Evaluating at ", step)
            model.eval()
            with torch.no_grad():
                num_batches = 0
                total_loss = 0
                predictions = None
                image_names = None
                for batch in test_loader:
                    image_test_names, images_test, ground_truth_3d = preprocess(batch, args, test_set)
                    prediction_3d = model(images_test, args).squeeze()
                    
                    for i, name in enumerate(image_test_names[:10]):
                        save_as_mesh(prediction_3d[i].unsqueeze(0), f"{args.debug_dir}/{step}_{name.item()}")
                        save_as_mesh(ground_truth_3d[i].unsqueeze(0), f"{args.debug_dir}/{step}_{name.item()}_gt")
                        cv2.imwrite(f"{args.debug_dir}/{step}_{name.item()}_gt.png", images_test[i].permute(0,2,3,1)[0].cpu().numpy())
                    
                    if predictions is None:
                        predictions = prediction_3d
                        image_names = image_test_names
                    else:
                        predictions = torch.cat([predictions, prediction_3d], dim=0)
                        image_names = torch.cat([image_names, image_test_names], dim=0)
                    
                    num_batches += 1
                    total_loss += calculate_voxel_loss(prediction_3d, ground_truth_3d.squeeze()).cpu().item()

                if num_batches != 0:
                    total_loss /= num_batches
                print("Loss at step {}: {}".format(step, total_loss))
                writer.add_scalar("eval_loss", total_loss, step)
                
                np.savetxt(os.path.join(args.logs_dir, dt, 'output_{:05d}_names.txt'.format(step)), image_names.cpu().numpy(), fmt="%d")
                output_file = h5py.File(os.path.join(args.logs_dir, dt, 'output_{:05d}.h5'.format(step)), 'w')
                output_file.create_dataset('tensor', data=predictions.cpu().numpy())
                output_file.close()
                
                # for i, name in enumerate(image_test_names):
                #     mesh = cubify(prediction_3d[i].unsqueeze(0), thresh=0.5)
                #     save_obj(f"{args.out_dir}/{step}_{name.item()}.obj", verts=mesh.verts_list()[0], faces=mesh.faces_list()[0])
                #     mesh_gt = cubify(ground_truth_3d[i].unsqueeze(0), thresh=0.5)
                #     save_obj(f"{args.out_dir}/{step}_{name.item()}_gt.obj", verts=mesh_gt.verts_list()[0], faces=mesh_gt.faces_list()[0])
                #     break
        model.train()
    print('Done!')
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
