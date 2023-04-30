import os
import argparse
from glob import glob

import cv2
import numpy as np

import torch

def parse_args():
    parser = argparse.ArgumentParser('dilate_thin_linesketches', add_help=False)
    
    parser.add_argument('--img_dir', default='../', type=str)
    parser.add_argument('--out_dir', default='../dilated', type=str)
    parser.add_argument('--img_type', default='jpeg', type=str)
    parser.add_argument('--resize', default=256, type=int)

    return parser.parse_args()

def main(args):
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    KERNEL = np.ones((3, 3), np.uint8)
    
    imgpaths = glob(f"{args.img_dir}/*.{args.img_type}")

    for imgpath in imgpaths:
        imgname = os.path.basename(imgpath)
        img = cv2.imread(imgpath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (args.resize, args.resize))
        
        eroded = cv2.erode(small, KERNEL, iterations=3)
        dilated = cv2.dilate(eroded, KERNEL, iterations=2)
        binary = cv2.threshold(dilated, 250, 255, cv2.THRESH_BINARY)[1]
        
        cv2.imwrite(f"{args.out_dir}/{imgname}", binary)
    
if __name__ == "__main__":
    args = parse_args()
    main(args)