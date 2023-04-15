import os
import sys
import time
import argparse
from tqdm import tqdm
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models as torchvision_models
import pytorch3d
import h5py

from model import Pix2Voxel
from train_utils import create_dir
from data_loader import IKEAManualStep

def parse_args():
    
    parser = argparse.ArgumentParser('IKEA23D', add_help=False)
    
    # Model parameters
    parser.add_argument('--encoder_arch', default='resnet18', type=str, help="architecture for encoder")
    parser.add_argument('--model', default='pix2vox', type=str, help="architecture for 3D reconstruction")
    
    # Training parameters
    parser.add_argument('--num_epochs', default=10000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--lr', default=4e-4, type=str)
    parser.add_argument('--train_test_split_ratio', default=0.7, type=float)
    parser.add_argument('--resize_w', default=224, type=int)
    parser.add_argument('--resize_h', default=224, type=int)
    parser.add_argument('--use_line_seg', action=argparse.BooleanOptionalAction)
    parser.add_argument('--use_seg_mask', action=argparse.BooleanOptionalAction)
    
    # Logging parameters
    parser.add_argument('--log_freq', default=1000, type=str)
    parser.add_argument('--save_freq', default=500, type=int)    
    parser.add_argument('--eval_freq', default=1000, type=int)
    
    parser.add_argument('--device', default='cuda', type=str) 
    
    # Directories & Checkpoint
    parser.add_argument('--load_checkpoint', default=None, type=str)            
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--logs_dir', type=str, default='./logs')
    parser.add_argument('--dataset_path', type=str, default='../dataset')
    
    return parser.parse_args()

def preprocess(feed_dict, args):
    #TODO: adjust the way data are loaded & formatted
    image_names = torch.tensor(feed_dict['image_num']).reshape(-1, 1)
    images = feed_dict['image'].squeeze(1)
    voxels = feed_dict['voxels'].float()
    ground_truth_3d = voxels
    
    return image_names.to(args.device), images.to(args.device), ground_truth_3d.to(args.device)

def calculate_voxel_loss(voxel_src, voxel_tgt):
    voxel_src = torch.clip(voxel_src, min=0, max=1)
    voxel_tgt = torch.clip(voxel_tgt, min=0, max=1)
    loss = torch.nn.BCELoss()(voxel_src, voxel_tgt)
    
    return loss
    
def main(args):
    ## Create directories for checkpoints and tensorboard logs
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.model)
    args.logs_dir = os.path.join(args.logs_dir, args.model)
    create_dir(args.checkpoint_dir)
    create_dir(args.logs_dir)
    
    ## Tensorboard Logger
    dt = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    create_dir(os.path.join(args.logs_dir, dt))
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
        start_iter = checkpoint['epoch']
        print(f"Succesfully loaded iter {start_iter}")
    
    ## Load dataset
    transform = transforms.Compose([
        transforms.Resize(tuple((args.resize_w, args.resize_h))),
        transforms.ToTensor()
    ]) #TODO: normalize?
    args.transforms = transform

    dataset = IKEAManualStep(args)
    train_set, test_set = torch.utils.data.random_split(dataset, [args.train_test_split_ratio, 1-args.train_test_split_ratio])

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
    
    print (f"Successfully loaded data: train_set {len(train_set)} | test_set {len(test_set)}")
    print("Starting training!")
    
    start_epoch = 0
    start_time = time.time()
    for epoch in range(start_epoch, args.num_epochs):
        epoch_start_time = time.time()

        if epoch % len(train_loader) == 0: #restart after one epoch
            train_loader = iter(train_dataloader)

        read_start_time = time.time()

        feed_dict = next(train_loader)

        _, images_gt, ground_truth_3d = preprocess(feed_dict, args)
        read_time = time.time() - read_start_time

        prediction_3d = model(images_gt, args).squeeze()

        loss = calculate_voxel_loss(prediction_3d, ground_truth_3d)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - epoch_start_time

        loss_vis = loss.cpu().item()

        if ((epoch + 1) % args.save_freq) == 0:
            print("saved at ", epoch)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(args.checkpoint_dir, 'checkpoint_{}.pth'.format(epoch)))

        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f" % (epoch, args.num_epochs, total_time, read_time, iter_time, loss_vis))

        if ((epoch + 1) % args.eval_freq) == 0:
            print("Evaluating at ", epoch)
            model.eval()
            with torch.no_grad():
                num_batches = 0
                total_loss = 0
                predictions = None
                image_names = None
                for batch in test_loader:
                    image_test_names, images_test, ground_truth_3d = preprocess(feed_dict, args)
                    prediction_3d = model(images_test, args).squeeze()
                    if predictions is None:
                        predictions = prediction_3d
                        image_names = image_test_names
                    else:
                        predictions = torch.cat([predictions, prediction_3d], dim=0)
                        image_names = torch.cat([image_names, image_test_names], dim=0)
                    num_batches += 1
                    total_loss += calculate_voxel_loss(prediction_3d, ground_truth_3d).cpu().item()

                if num_batches != 0:
                    total_loss /= num_batches
                print("Loss at epoch {}: {}".format(epoch, total_loss))
                np.savetxt(os.path.join(args.logs_dir, dt, 'output_{:05d}_names.txt'.format(epoch)), image_names.cpu().numpy(), fmt="%d")
                output_file = h5py.File(os.path.join(args.logs_dir, dt, 'output_{:05d}.h5'.format(epoch)), 'w')
                output_file.create_dataset('tensor', data=predictions.cpu().numpy())
                output_file.close()
        model.train()
    print('Done!')
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
