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

from model import Pix2Voxel
from utils import create_dir
from data_loader import IKEAManual

def parse_args():
    
    parser = argparse.ArgumentParser('IKEA23D', add_help=False)
    
    # Model parameters
    parser.add_argument('--encoder_arch', default='resnet18', type=str, help="architecture for encoder")
    parser.add_argument('--model', default='pix2vox', type=str, help="architecture for 3D reconstruction")
    
    # Training parameters
    parser.add_argument('--num_epochs', default=10000, type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=0, type=str)
    parser.add_argument('--lr', default=4e-4, type=str)
    parser.add_argument('--train_test_split_ratio', default=0.7, type=float)
    
    # Logging parameters
    parser.add_argument('--log_freq', default=1000, type=str)
    parser.add_argument('--save_freq', default=500, type=int)    
    
    parser.add_argument('--device', default='cuda', type=str) 
    
    # Directories & Checkpoint
    parser.add_argument('--load_feat', action='store_true') 
    parser.add_argument('--load_checkpoint', default='', type=str)            
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--logs_dir', type=str, default='./logs')
    parser.add_argument('--data_dir', type=str, default='./dataset/ikea_man')
    
    return parser

def preprocess(feed_dict, args):
    #TODO: adjust the way data are loaded & formatted
    images = feed_dict['images'].squeeze(1)
    voxels = feed_dict['voxels'].float()
    ground_truth_3d = voxels
    
    if args.load_feat:
        feats = torch.stack(feed_dict['feats'])
        return feats.to(args.device), ground_truth_3d.to(args.device)
    else:
        return images.to(args.device), ground_truth_3d.to(args.device)

def calculate_loss(voxel_src, voxel_tgt):
    voxel_src = torch.clip(voxel_src, min=0, max=1)
    voxel_tgt = torch.clip(voxel_tgt, min=0, max=1)
    loss = torch.nn.BCELoss()(voxel_src, voxel_tgt)
    
    return loss
    
def main(args):
    
    ## Create directories for checkpoints and tensorboard logs
    args.checkpoint_dir = f"{args.checkpoint_dir}/{args.model}"
    args.logs_dir = f"{args.logs_dir}/{args.model}"
    create_dir(args.checkpoint_dir)
    create_dir(args.logs_dir)
    
    ## Tensorboard Logger
    dt = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    create_dir(f'{args.log_dir}/{dt}')
    writer = SummaryWriter(f'{args.log_dir}/{dt}')
    
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
    dataset = IKEAManual(args.data_dir)
    train_set, val_set = torch.utils.data.random_split(dataset, [args.train_test_split_ratio, 1-args.train_test_split_ratio])
    train_dataloader = DataLoader(dataset=train_set, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              num_workers=args.num_workers)
    val_loader = DataLoader(dataset=val_set, 
                            batch_size=args.batch_size, 
                            shuffle=True, 
                            num_workers=args.num_workers)
    train_loader = iter(train_dataloader)
    
    print (f"Successfully loaded data: train_set {len(train_set)} | val_set {len(val_set)}")
    
    print("Starting training!")
    # for epoch in range(args.num_epochs):
        
    #     ## Train
    #     train_loss = train(train_loader, model, optimizer, epoch, args, writer)
        
    #     ## Evaluate
    #     val_loss = validate(val_loader, model, epoch, args, writer)
        
    #     print ("epoch: {}   train loss: {:.4f}   val loss: {:.4f}".format(epoch, train_loss, val_loss))
        
    #     if epoch % args.checkpoint_every == 0:
    #         print ("checkpoint saved at epoch {}".format(epoch))
    #         save_checkpoint(epoch=epoch, model=model, args=args, best=False)
    
    start_epoch = 0
    start_time = time.time()
    for epoch in range(start_epoch, args.num_epochs):
        epoch_start_time = time.time()

        if epoch % len(train_loader) == 0: #restart after one epoch
            train_loader = iter(train_dataloader)

        read_start_time = time.time()

        feed_dict = next(train_loader)

        images_gt, ground_truth_3d = preprocess(feed_dict, args)
        read_time = time.time() - read_start_time

        prediction_3d = model(images_gt, args)

        loss = calculate_loss(prediction_3d, ground_truth_3d)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - epoch_start_time

        loss_vis = loss.cpu().item()

        if (epoch % args.save_freq) == 0:
            print("saved at ", epoch)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, f'checkpoint_{epoch}.pth')

        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f" % (epoch, args.num_epochs, total_time, read_time, iter_time, loss_vis))

    print('Done!')
    

if __name__ == "__main__":
    args = parse_args()
    main(args)