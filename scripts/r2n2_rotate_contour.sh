#!/bin/sh

python img2line.py --img_dir ../dataset/r2n2_shapenet_dataset/r2n2 \
                    --out_dir ../dataset/shapenet_rotate \
                    --rotate

# python img2line.py --img_dir ../dataset/shapenet_rotate \
#                     --out_dir ../dataset/shapenet_rotate_edge