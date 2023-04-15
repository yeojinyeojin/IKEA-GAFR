import os
import sys
import math
import argparse
from glob import glob

import cv2
import torch
from pytorch3d.datasets.r2n2.utils import (
    BlenderCamera,
    align_bbox,
    compute_extrinsic_matrix,
    read_binvox_coords,
    voxelize,
)
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj, save_obj
from pytorch3d.transforms import euler_angles_to_matrix, Rotate
from pytorch3d.renderer import TexturesVertex, look_at_view_transform
from pytorch3d.renderer.cameras import FoVPerspectiveCameras
from pytorch3d.vis.plotly_vis import plot_scene

def parse_args():
    
    parser = argparse.ArgumentParser('Image2LineDrawing', add_help=False)
    
    parser.add_argument('--img_dir', type=str, default='../dataset/r2n2_shapenet_dataset/r2n2')
    parser.add_argument('--vox_dir', type=str, default='../dataset/r2n2_shapenet_dataset/shapenet')
    parser.add_argument('--out_dir', type=str, default='../dataset/r2n2_shapenet_line')
    
    parser.add_argument('--sobel', action='store_true', default=False)
    parser.add_argument('--rotate', action='store_true', default=True)
    
    return parser.parse_args()

def rotate(datadir, outdir):
    ROT_ANGLES = [30, 60, 90]
    voxpaths = glob(f"{datadir}/**/*.obj", recursive=True)
    
    for voxpath in voxpaths:
        obj_name = voxpath.split('/')[-2]
        
        verts, faces, _ = load_obj(voxpath)
        faces = faces.verts_idx
        
        for angle in ROT_ANGLES:
            rot_mat = euler_angles_to_matrix(math.radians(angle))
            R = Rotate(rot_mat)
            verts_rot = R.transform_points(verts)
            
            textures = torch.ones_like(verts_rot)*torch.tensor([[0.0, 1.0, 0.0]])
            mesh = Meshes(
                    verts=verts_rot.unsqueeze(0),
                    faces=faces.verts_idx.unsqueeze(0),
                    textures=TexturesVertex(textures.unsqueeze(0))
                )
            output_path = f"{outdir}/{obj_name}.obj"
            save_obj(f=output_path, verts=verts, faces=faces)

    return outdir
    
def get_contour_and_save(imgdir, rot=None):
    imgpaths = glob(f"{imgdir}/**/*.png", recursive=True)
    
    for imgpath in imgpaths:
        obj_name = imgpath.split('/')[-3]
        order = imgpath.split('/')[-1]
        
        img = cv2.imread(imgpath)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, ksize=(3,3), sigmaX=cv2.BORDER_DEFAULT)
        
        if args.sobel:
            sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
            sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
            sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
            cv2.imwrite(f"{args.out_dir}/{obj_name}_{order}_sobel.png", sobelxy)
        
        canny = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
        canny_inv = cv2.bitwise_not(canny)
        cv2.imwrite(f"{args.out_dir}/{obj_name}_{order}_{rot}.png", canny_inv)

def main(args):
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    if args.rotate:
        rootdir = os.path.abspath(os.path.join(args.out_dir, os.pardir))
        datasetname = os.path.basename(args.vox_dir)
        rotdir = f"{rootdir}/{datasetname}_rotate"
        os.makedirs(rotdir, exist_ok=True)
        
        imgdir = rotate(args.vox_dir, rotdir)
    else:
        imgdir = args.img_dir
    
    get_contour_and_save(imgdir, rot=None)
    

if __name__ == "__main__":
    args = parse_args()
    main(args)