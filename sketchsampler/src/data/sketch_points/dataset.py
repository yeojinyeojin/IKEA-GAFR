import os
import pickle as pkl
from typing import Any

import cv2
import numpy as np
from common.utils import rgb2gray
from omegaconf import ValueNode, DictConfig
from skimage import io
from torch.utils.data import Dataset
import torch

class DataSet(Dataset):
    def __init__(self, name: ValueNode, path_sketch: str, path_pt: str, path_camera: str,
                 path_density: str, file_list: str, cfg: DictConfig, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.name = name

        self.path_sketch = path_sketch
        self.path_pt = path_pt
        self.path_camera = path_camera
        self.path_density = path_density
        self.pkl_list = []
        with open(file_list, 'r') as f:
            while (True):
                line = f.readline().strip()
                if not line:
                    break
                self.pkl_list.append(line)

    def __len__(self):
        return len(self.pkl_list)

    def __getitem__(self, idx):
        camera_path = os.path.join(self.path_camera, self.pkl_list[idx])
        density_path = os.path.join(self.path_density, self.pkl_list[idx])
        pointcloud_path = os.path.join(self.path_pt, "/".join(self.pkl_list[idx].split('/')[:2] + ["pt.dat"]))
        
        raw_pointcloud = pkl.load(open(pointcloud_path, 'rb'))
        cam_mat, cam_pos = pkl.load(open(camera_path, 'rb'))
        
        density_map = pkl.load(open(density_path, 'rb')).astype('float32')
        density_map /= np.sum(density_map)
        density_map = density_map[None, ...]
        
        gt_pointcloud = np.dot(raw_pointcloud - cam_pos, cam_mat.transpose())
        gt_pointcloud[:, 2] -= np.mean(gt_pointcloud[:, 2])
        
        sketch_path = os.path.join(self.path_sketch, self.pkl_list[idx].replace('.dat', '.png'))
        sketch = io.imread(sketch_path)
        sketch[np.where(sketch[:, :, 3] == 0)] = 255
        sketch = sketch.astype('float32') / 255
        sketch = ((np.transpose(rgb2gray(sketch[:, :, :3]), (2, 0, 1)) - .5) * 2).astype('float32')
        
        metadata = self.pkl_list[idx][:-4]
        
        item = (sketch, gt_pointcloud, density_map, metadata)
        return item

    def __repr__(self):
        return f"Dataset(name={self.name!r}"
    
class CustomDataSet(Dataset):
    def __init__(self, name: ValueNode, path_sketch: str, path_pt: str, path_camera: str,
                 path_density: str, file_list: str, cfg: DictConfig, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.name = name

        self.path_sketch = path_sketch
        self.path_pt = path_pt
        self.path_camera = path_camera
        self.path_density_seg = path_density
        self.pkl_list = []
        with open(file_list, 'r') as f:
            while (True):
                line = f.readline().strip()
                if not line:
                    break
                self.pkl_list.append(line)

    def __len__(self):
        return len(self.pkl_list)

    def __getitem__(self, idx):
        # breakpoint()
        camera_path = os.path.join(self.path_camera, self.pkl_list[idx])
        seg_density_path = os.path.join(self.path_density_seg, self.pkl_list[idx])
        pointcloud_path = os.path.join(self.path_pt, "/".join(self.pkl_list[idx].split('/')[:2] + ["pt.dat"]))
        
        raw_pointcloud = pkl.load(open(pointcloud_path, 'rb'))
        cam_mat, cam_pos = pkl.load(open(camera_path, 'rb'))
        
        # density_map = pkl.load(open(density_path, 'rb')).astype('float32')
        # density_map /= np.sum(density_map)
        # density_map = density_map[None, ...]
        seg_density_map = pkl.load(open(seg_density_path, 'rb')).astype('float32')
        
        ## use canonical view instead of object-centric view
        gt_pointcloud = raw_pointcloud
        # gt_pointcloud = np.dot(raw_pointcloud - cam_pos, cam_mat.transpose())
        # gt_pointcloud[:, 2] -= np.mean(gt_pointcloud[:, 2])
        
        sketch_path = os.path.join(self.path_sketch, self.pkl_list[idx].replace('.dat', '.png'))
        sketch = io.imread(sketch_path)
        sketch[np.where(sketch[:, :, 3] == 0)] = 255
        sketch = sketch.astype('float32') / 255
        sketch = ((np.transpose(rgb2gray(sketch[:, :, :3]), (2, 0, 1)) - .5) * 2).astype('float32')
        
        metadata = self.pkl_list[idx][:-4]
        
        item = (sketch, gt_pointcloud, seg_density_map, metadata)
        return item

    def __repr__(self):
        return f"Dataset(name={self.name!r}"


    
class CustomDataSet2(Dataset):
    def __init__(self, name: ValueNode, path_sketch: str, path_pt: str, path_camera: str,
                 path_density: str, file_list: str, cfg: DictConfig, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.name = name

        self.dims = (256, 256)
        self.path_label = path_camera
        self.path_sketch = path_sketch
        self.path_pt = path_pt
        # self.path_camera = path_camera
        self.path_density_seg = path_density
        
        # self.pkl_list = []
        # with open(file_list, 'r') as f:
        #     while (True):
        #         line = f.readline().strip()
        #         if not line:
        #             break
        #         self.pkl_list.append(line)
        # breakpoint()

    def __len__(self):
        return len(os.listdir(self.path_density_seg))

    def __getitem__(self, idx):
        # breakpoint()
        # camera_path = os.path.join(self.path_camera, self.pkl_list[idx])
        seg_density_path = f"{self.path_density_seg}/{str(idx).zfill(5)}.npy"
        
        pointcloud_path = f"{self.path_pt}/{str(idx).zfill(5)}_points.npy"
        label_path = f"{self.path_label}/{str(idx).zfill(5)}_labels.npy"
        
        raw_pointcloud = np.load(pointcloud_path)
        # cam_mat, cam_pos = pkl.load(open(camera_path, 'rb'))
        
        # density_map = pkl.load(open(density_path, 'rb')).astype('float32')
        # density_map /= np.sum(density_map)
        # density_map = density_map[None, ...]
        seg_density_map = np.load(seg_density_path).astype('float')
        # seg_density_map = torch.tensor(seg_density_map).permute(2,0,1)
        seg_density_map = np.transpose(seg_density_map, (2,0,1))
        
        # drop the first channel due to artefacts from interpolation
        seg_density_map = seg_density_map[1:]
        
        total_m = np.sum(seg_density_map, axis=0)
        
        # todo: refactor later
        _denom = np.sum(seg_density_map, axis=(0), keepdims=True)
        nonzero_idxs = _denom.nonzero()
        channel_m = seg_density_map.copy()
        for _i in range(len(nonzero_idxs)):
            y, x = nonzero_idxs[1][_i], nonzero_idxs[2][_i]
            channel_m[:, y, x] = channel_m[:, y, x] / _denom[:, y, x]
        total_m = total_m / np.sum(total_m)
        
        # channel_m = np.divide(seg_density_map, _denom, where=_denom!=0)
        # channel_m = seg_density_map / np.sum(seg_density_map, axis=(0), keepdims=True)  # [4, 24, 256, 256]
        # channel_m = np.nan_to_num(channel_m)
        
        seg_density_map = np.concatenate((np.expand_dims(total_m, axis=0), channel_m), axis=0)
        

        label = np.load(label_path)
        
        ## use canonical view instead of object-centric view
        gt_pointcloud = raw_pointcloud
        # gt_pointcloud = np.dot(raw_pointcloud - cam_pos, cam_mat.transpose())
        # gt_pointcloud[:, 2] -= np.mean(gt_pointcloud[:, 2])
        
        sketch_path = f"{self.path_sketch}/{str(idx).zfill(5)}.png"
        # sketch_path = os.path.join(self.path_sketch, self.pkl_list[idx].replace('.dat', '.png'))
        sketch = io.imread(sketch_path)
        sketch = cv2.resize(sketch, self.dims, interpolation = cv2.INTER_AREA)
        # sketch[np.where(sketch[:, :, 3] == 0)] = 255
        sketch = sketch.astype('float32') / 255
        sketch = ((np.transpose(rgb2gray(sketch[:, :, :3]), (2, 0, 1)) - .5) * 2).astype('float32')
        # metadata = self.pkl_list[idx][:-4]

        item = (sketch, gt_pointcloud, seg_density_map, label)
        # item = (sketch, gt_pointcloud, seg_density_map, metadata)
        return item

    def __repr__(self):
        return f"Dataset(name={self.name!r}"
