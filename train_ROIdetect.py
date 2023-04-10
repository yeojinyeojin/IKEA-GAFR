import os
import sys
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision import models as torchvision_models

from train_utils import create_dir, ComputeLoss
from data_loader import IKEAManual


def create_args():
    parser = argparse.ArgumentParser('RoI_Detection', add_help=False)
    
    # Model parameters
    parser.add_argument('--model', default='yolo', type=str, help="architecture for detection")
    
    # Training parameters
    parser.add_argument('--num_epochs', default=10000, type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=0, type=str)
    parser.add_argument('--lr', default=4e-4, type=str)
    parser.add_argument('--train_test_split_ratio', default=0.7, type=float)
    parser.add_argument('--resize_w', default=640, type=int)
    parser.add_argument('--resize_h', default=480, type=int)
    
    # Logging parameters
    parser.add_argument('--log_freq', default=1000, type=str)
    parser.add_argument('--save_freq', default=500, type=int)    
    
    parser.add_argument('--device', default='cuda', type=str) 
    
    # Directories & Checkpoint
    parser.add_argument('--load_feat', action='store_true') 
    parser.add_argument('--load_checkpoint', default=None, type=str)            
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/RoI_detect')
    parser.add_argument('--logs_dir', type=str, default='./logs/RoI_detect')
    parser.add_argument('--data_dir', type=str, default='./dataset')
    
    return parser.parse_args()

# def compute_loss(pred, gt):
#     pass

def main(args):
    
    ## Create directories for checkpoints and tensorboard logs
    args.checkpoint_dir = f"{args.checkpoint_dir}/{args.model}"
    args.logs_dir = f"{args.logs_dir}/{args.model}"
    create_dir(args.checkpoint_dir)
    create_dir(args.logs_dir)
    
    ## Tensorboard Logger
    dt = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    create_dir(f'{args.logs_dir}/{dt}')
    writer = SummaryWriter(f'{args.logs_dir}/{dt}')
    
    ## Load model & optimizer
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False, pretrained=False)
    model = model.to(args.device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    compute_loss = ComputeLoss(model)
    
    ## Load dataset
    transform = transforms.Compose([
        transforms.Resize(tuple((args.resize_w, args.resize_h))),
        transforms.ToTensor()
        
    ]) #TODO: normalize?
    dataset = IKEAManual(args.data_dir, args.load_feat, transform)
    train_set, val_set = torch.utils.data.random_split(dataset, [args.train_test_split_ratio, 1-args.train_test_split_ratio])
    train_dataloader = DataLoader(dataset=train_set, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              num_workers=args.num_workers)
    val_dataloader = DataLoader(dataset=val_set, 
                            batch_size=args.batch_size, 
                            shuffle=True, 
                            num_workers=args.num_workers)
    
    print (f"Successfully loaded data: train_set {len(train_set)} | val_set {len(val_set)}")
    
    for epoch in range(args.num_epochs):
        for batch_i, data_dict in enumerate(train_dataloader):
            imgs = data_dict['images'].to(args.device)
            targets = data_dict['labels'].to(args.device)

            optimizer.zero_grad()

            prediction = model(imgs)
            loss, loss_items = compute_loss(prediction, targets)

            loss.backward()
            optimizer.step()
            
            writer.add_scalar('train_loss', loss.item(), epoch)

            print(
                "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
                % (
                    epoch,
                    args.num_epochs,
                    batch_i,
                    len(train_dataloader),
                    model.losses["x"],
                    model.losses["y"],
                    model.losses["w"],
                    model.losses["h"],
                    model.losses["conf"],
                    model.losses["cls"],
                    loss.item(),
                    model.losses["recall"],
                    model.losses["precision"],
                )
            )

            model.seen += imgs.size(0)

        if epoch % args.save_freq == 0:
            print("saved at ", epoch)
            # model.save_weights("%s/%d.weights" % (args.checkpoint_dir, epoch))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, f'checkpoint_{epoch}.pth')

if __name__ == "__main__":
    args = create_args()
    main(args)