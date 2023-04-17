import os
import sys
import math
import argparse
from glob import glob

import cv2
import torch
import numpy as np
from tqdm import tqdm
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
from pytorch3d.renderer import (
    TexturesVertex, 
    look_at_view_transform, 
    FoVPerspectiveCameras, 
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader
)
from pytorch3d.renderer.cameras import look_at_view_transform
from pytorch3d.vis.plotly_vis import plot_scene

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    
    parser = argparse.ArgumentParser('Image2LineDrawing', add_help=False)
    
    # parser.add_argument('--img_dir', type=str, default='../dataset/shapenet_rotate')
    parser.add_argument('--img_dir', type=str, default='../dataset/r2n2_shapenet_dataset/r2n2/ShapeNetRendering/03001627')
    # parser.add_argument('--img_dir', type=str, default='../dataset/r2n2_shapenet_dataset/r2n2')
    parser.add_argument('--vox_dir', type=str, default='../dataset/r2n2_shapenet_dataset/shapenet')
    # parser.add_argument('--out_dir', type=str, default='../dataset/shapenet_rotate_edge')
    # parser.add_argument('--out_dir', type=str, default='../dataset/r2n2_shapenet_rotate_line')
    parser.add_argument('--out_dir', type=str, default='../dataset/r2n2_shapenet_original_line')
    
    parser.add_argument('--sobel', action='store_true', default=False)
    parser.add_argument('--rotate', action='store_true', default=False)
    parser.add_argument('--num_rotate', type=int, default=10)
    parser.add_argument('--visualize', action='store_true', default=False)
    
    return parser.parse_args()

def get_mesh_renderer(image_size=512, lights=None, device=None):
    """
    Returns a Pytorch3D Mesh Renderer.

    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer

def rotate(datadir, outdir, num_rotate=10, visualize=False):
    voxpaths = glob(f"{datadir}/**/*.obj", recursive=True)
    
    lights = PointLights(location=[[0, 0, -3]])
    renderer = get_mesh_renderer(image_size=137)
    
    angles = torch.bernoulli(torch.rand(num_rotate*len(voxpaths), 3))
    angles *= torch.randint(0, 360, (num_rotate*len(voxpaths), 3))
    angles = angles.int()
    
    for i, voxpath in tqdm(enumerate(voxpaths), total=len(voxpaths)):
        obj_name = voxpath.split('/')[-2]
        
        os.makedirs(f"{outdir}/{obj_name}", exist_ok=True)
        
        # if len(glob(f"home/ubuntu/IKEA/dataset/shapenet_rotate/{obj_name}*.png")) == 5:
        #     continue
        
        # verts, faces, _ = load_obj(voxpath)
        try:
            verts, faces, _ = load_obj(voxpath)
        except:
            continue
        faces = faces.verts_idx
        
        for angle in angles[i*num_rotate:i*num_rotate+num_rotate]:
            # euler = torch.Tensor([0, 0, 90]) / 180 * torch.pi
            euler = angle / 180 * torch.pi
            rot_mat = euler_angles_to_matrix(euler, "XYZ")
            R = Rotate(rot_mat)
            verts_rot = R.transform_points(verts)
            
            # output_path = f"{outdir}/{obj_name}.obj"
            # save_obj(f=output_path, verts=verts_rot, faces=faces)
            
            cameras = FoVPerspectiveCameras(
                R=torch.eye(3).unsqueeze(0),
                T=torch.tensor([[0, 0, 1]]), 
                fov=60
                )
            textures = torch.ones_like(verts_rot) * torch.tensor([[1.0, 0.0, 0.0]])
            mesh = Meshes(
                    verts=verts_rot.unsqueeze(0),
                    faces=faces.unsqueeze(0),
                    textures=TexturesVertex(textures.unsqueeze(0))
                )
            rend = renderer(mesh.cuda(), cameras=cameras.cuda(), lights=lights.cuda())
            rend = rend.detach().cpu().numpy()[0, ..., :3]
            rend *= 255
            
            cv2.imwrite(f"{outdir}/{obj_name}/x{angle[0]}_y{angle[1]}_z{angle[2]}.png", rend)
            
            if visualize:
                
                gt_vox_verts, gt_vox_face_data, _ = load_obj(voxpath)
                gt_vox_verts += torch.tensor([[1.0, 0.0, 0.0]])
                gt_textures = torch.ones_like(gt_vox_verts) * torch.tensor([[0.0, 1.0, 0.0]])
                gt_mesh = Meshes(
                    verts=gt_vox_verts.unsqueeze(0),
                    faces=gt_vox_face_data.verts_idx.unsqueeze(0),
                    textures=TexturesVertex(gt_textures.unsqueeze(0))
                )
                
                scene = plot_scene({
                    "Figure1": {
                        "Rotated Mesh": mesh,
                        "GT Mesh": gt_mesh
                    }
                })
                scene.show()
                input("Close viz window and press enter to continue to next sample...")

    return outdir
    
def get_contour_and_save(imgdir, args):
    dirpaths = glob(f"{imgdir}/*", recursive=True)
    # imgpaths = glob(f"{imgdir}/**/*.png", recursive=True)
    
    # for imgpath in tqdm(imgpaths, total=len(imgpaths)):
    #     if 'r2n2' in os.path.basename(imgdir):
    #         obj_name = imgpath.split('/')[-3]
    #         order = imgpath.split('/')[-1][:-4]
    #     elif 'shapenet' in os.path.basename(imgdir):
    #         obj_name = imgpath.split('/')[-2]
    #         order =  imgpath.split('/')[-1][:-4]
    #         # obj_name = os.path.basename(imgpath).split('_')[0]
    #         # order = os.path.basename(imgpath).replace(obj_name, '')[1:-4]
        # obj_outdir = os.path.join(args.out_dir, obj_name)
        # os.makedirs(obj_outdir, exist_ok=True)
        
        # img = cv2.imread(imgpath)
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img_blur = cv2.GaussianBlur(img_gray, ksize=(3,3), sigmaX=cv2.BORDER_DEFAULT)
        
        # if args.sobel:
        #     sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
        #     sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
        #     sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
            
        #     cv2.imwrite(os.path.join(obj_outdir, f"{order}_sobel.png"), sobelxy)
        
        # canny = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
        # canny_inv = cv2.bitwise_not(canny)
        
        # cv2.imwrite(os.path.join(obj_outdir, f"{order}.png"), canny_inv)
    
    skip_cnt = 0
    for dirpath in tqdm(dirpaths, total=len(dirpaths)):
        
        files = glob(f"{dirpath}/**/*.png", recursive=True)
        if len(files) < 5 :
            skip_cnt += 1
            continue
        
        # if 'shapenet' in os.path.basename(imgdir):
        obj_name = os.path.basename(os.path.normpath(dirpath))
        
        obj_outdir = os.path.join(args.out_dir, obj_name)
        os.makedirs(obj_outdir, exist_ok=True)
        
        if args.rotate:
            files = files[:5]
        
        for imgpath in files:
            order = os.path.basename(imgpath)[:-4]
            
            img = cv2.imread(imgpath)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_blur = cv2.GaussianBlur(img_gray, ksize=(3,3), sigmaX=cv2.BORDER_DEFAULT)
            
            if args.sobel:
                sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
                sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
                sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
                
                cv2.imwrite(os.path.join(obj_outdir, f"{order}_sobel.png"), sobelxy)
            
            canny = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
            canny_inv = cv2.bitwise_not(canny)
            
            cv2.imwrite(os.path.join(obj_outdir, f"{order}.png"), canny_inv)
    print("@@@ skip_cnt: ", skip_cnt)

def main(args):
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    if args.rotate:
        rootdir = os.path.abspath(os.path.join(args.out_dir, os.pardir))
        datasetname = os.path.basename(args.vox_dir)
        rotdir = f"{rootdir}/{datasetname}_rotate"
        os.makedirs(rotdir, exist_ok=True)
        
        imgdir = rotate(args.vox_dir, rotdir, args.num_rotate, args.visualize)
    else:
        imgdir = args.img_dir
    
    get_contour_and_save(imgdir, args)
    

if __name__ == "__main__":
    args = parse_args()
    main(args)