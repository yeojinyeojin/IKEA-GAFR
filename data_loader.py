import os
import sys
import argparse
from glob import glob
import json
import h5py
import random
import warnings
from pathlib import Path
from tabulate import tabulate
from collections import defaultdict

import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision import models as torchvision_models
from pytorch3d.datasets.shapenet_base import ShapeNetBase
from pytorch3d.renderer import HardPhongShader
from pytorch3d.datasets.r2n2 import utils
from pytorch3d.datasets.r2n2.utils import (
    BlenderCamera,
    align_bbox,
    compute_extrinsic_matrix,
    read_binvox_coords,
    voxelize,
)

import utils_vox

def read_hdf5(file, key = 'tensor'):
    assert os.path.exists(file), 'file %s not found' % file
    h5f = h5py.File(file, 'r')
    assert key in h5f.keys(), 'key %s not found in file %s' % (key, file)
    gt_voxels = torch.from_numpy(h5f[key][()])
    h5f.close()

    return gt_voxels

# class IKEAManualPDF(Dataset): #currently not using (instead, train YOLOv5 as is)
#     def __init__(self, datadir, transforms=None):
#         super().__init__()
        
#         self.imgdir = f"{datadir}/images"
#         self.imgpaths = glob(f"{self.imgdir}/**/*.png", recursive=True)
        
        
#         self.annotfile = f"{datadir}/annotation.json" #TODO
#         with open(self.annotfile) as f:
#             self.annotation = json.load(f)
        
#         self.imgs = []
#         for imgpath in self.imgpaths:
#             img = Image.open(imgpath)
#             if transforms is not None:
#                 img = transforms(img)
#             self.imgs.append(img)
#         # sorted(self.imgs) #TODO: needed?
        
#         #TODO: load labels
#         self.labels = torch.randn(len(self.imgs), 5)
#         # self.labels = []
        
#     def __len__(self):
#         return len(self.imgs)

#     def __getitem__(self, idx):
#         dic = {
#             'images': self.imgs[idx],
#             'voxels': self.voxels[idx],
#         }
        
#         return dic

class IKEAManualStep(Dataset):
    def __init__(self, args, chairs_only=False):
        super().__init__()

        datadir = args.dataset_path
        transforms = args.transforms

        self.img_dir = os.path.join(datadir, "images", "rgb") #Change to images/rgb/ later
        self.img_paths = sorted(glob(os.path.join(self.img_dir, "**", "*.png"), recursive=True))
        self.img_metadata = json.load(open(os.path.join(datadir, "ind_map.json"))) #Gives image height/width and other useful info

        self.label_dir = os.path.join(datadir, "labels")
        self.label_paths = sorted(glob(os.path.join(self.label_dir, "**", "*.txt"), recursive=True))

        if args.use_line_seg: #If we want to concatenate line segmentations to input
            self.line_seg_dir = os.path.join(datadir, "images", "line_seg")
            self.line_seg_paths = sorted(glob(os.path.join(self.line_seg_dir, "**", "*.png"), recursive=True)) 
        
        if args.use_seg_mask: #If we want to concatenate segmentation masks to input
            self.seg_mask_dir = os.path.join(datadir, "images", "mask")
            self.seg_mask_paths = sorted(glob(os.path.join(self.seg_mask_dir, "**", "*.png"), recursive=True))


        self.gt_voxels = read_hdf5(os.path.join(datadir, "gt_off_32_x_32", "output.h5"))

        self.imgs = []
        self.img_nums = []

        for imgpath, labelpath in zip(self.img_paths, self.label_paths):
            print(imgpath, labelpath)
            imgpath = imgpath.split("/")[-1]
            labelpath = labelpath.split("/")[-1]

            img = cv2.imread((os.path.join(self.img_dir, imgpath)))


            if args.use_line_seg:
                line_seg_img = cv2.imread(os.path.join(self.line_seg_dir, imgpath))
                img = np.dstack([img, line_seg_img]) #Stack RGB img and line segmentation depth-wise
            if args.use_seg_mask:
                seg_mask_img = cv2.imread(os.path.join(self.seg_mask_dir, imgpath))
                img = np.dstack([img, seg_mask_img]) #Stack RGB img and segmentation mask depth-wise

            img_data = self.img_metadata[imgpath]
            h, w = img_data["img_h"], img_data["img_w"]

            with open(os.path.join(self.label_dir, labelpath)) as f: #Read in label txt file for specific image
                for line in f.readlines(): #For each bbox in image
                    line = np.array(line.split(" ")).astype(np.float32).reshape(-1) # [object_class, relevancy_class, x_c, y_c, w, h]
                    line = torch.from_numpy(line)

                    category = line[0]
                    relevancy = line[1]

                    if relevancy == 1: #If bounded region is relevant
                        bbox_x_center = line[2]
                        bbox_y_center = line[3]
                        bbox_w = line[4]
                        bbox_h = line[5]

                        #Convert x_c, y_c, w, h to start_x, start_y, end_x, end_y 
                        start_x = torch.floor(w*(bbox_x_center - bbox_w/2)).to(int).item()
                        start_y = torch.floor(h*(bbox_y_center - bbox_h/2)).to(int).item()

                        end_x = torch.floor(w*(bbox_x_center + bbox_w/2)).to(int).item()
                        end_y = torch.floor(h*(bbox_y_center + bbox_h/2)).to(int).item()

                        #Get bounded region
                        cropped_img = img[start_y: end_y, start_x: end_x]
                        
                        #Convert to PIL image for PyTorch transforms
                        cropped_img = Image.fromarray(cropped_img)

                        #Apply transforms if they exist
                        if transforms is not None:
                            cropped_img = transforms(cropped_img)

                        self.img_nums.append(int(imgpath.split(".")[0]))
                        self.imgs.append(cropped_img)
    
        assert len(self.img_nums) == len(self.imgs)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        dic = {
            'names': self.img_nums[idx],
            'images': self.imgs[idx],
            'voxels': self.gt_voxels[idx],
        }
        
        return dic

SYNSET_DICT_DIR = Path(utils.__file__).resolve().parent
MAX_CAMERA_DISTANCE = 1.75  # Constant from R2N2.
VOXEL_SIZE = 128
# Intrinsic matrix extracted from Blender. Taken from meshrcnn codebase:
# https://github.com/facebookresearch/meshrcnn/blob/main/shapenet/utils/coords.py
BLENDER_INTRINSIC = torch.tensor(
    [
        [2.1875, 0.0, 0.0, 0.0],
        [0.0, 2.1875, 0.0, 0.0],
        [0.0, 0.0, -1.002002, -0.2002002],
        [0.0, 0.0, -1.0, 0.0],
    ]
)

class R2N2_orig(ShapeNetBase):  # pragma: no cover
    """
    This class loads the R2N2 dataset from a given directory into a Dataset object.
    The R2N2 dataset contains 13 categories that are a subset of the ShapeNetCore v.1
    dataset. The R2N2 dataset also contains its own 24 renderings of each object and
    voxelized models. Most of the models have all 24 views in the same split, but there
    are eight of them that divide their views between train and test splits.

    """

    def __init__(
        self,
        split: str,
        shapenet_dir,
        r2n2_dir,
        splits_file,
        return_all_views: bool = True,
        return_voxels: bool = False,
        return_feats: bool = False,
        return_RTK: bool = False,
        views_rel_path: str = "ShapeNetRendering",
        voxels_rel_path: str = "ShapeNetVoxels",
        load_textures: bool = False,
        texture_resolution: int = 4,
    ):
        """
        Store each object's synset id and models id the given directories.

        Args:
            split (str): One of (train, val, test).
            shapenet_dir (path): Path to ShapeNet core v1.
            r2n2_dir (path): Path to the R2N2 dataset.
            splits_file (path): File containing the train/val/test splits.
            return_all_views (bool): Indicator of whether or not to load all the views in
                the split. If set to False, one of the views in the split will be randomly
                selected and loaded.
            return_voxels(bool): Indicator of whether or not to return voxels as a tensor
                of shape (D, D, D) where D is the number of voxels along each dimension.
            return_feats(bool): Indicator of whether image features from a pretrained resnet18 
                are also returned in the dataloader or not
            views_rel_path: path to rendered views within the r2n2_dir. If not specified,
                the renderings are assumed to be at os.path.join(rn2n_dir, "ShapeNetRendering").
            voxels_rel_path: path to rendered views within the r2n2_dir. If not specified,
                the renderings are assumed to be at os.path.join(rn2n_dir, "ShapeNetVoxels").
            load_textures: Boolean indicating whether textures should loaded for the model.
                Textures will be of type TexturesAtlas i.e. a texture map per face.
            texture_resolution: Int specifying the resolution of the texture map per face
                created using the textures in the obj file. A
                (texture_resolution, texture_resolution, 3) map is created per face.

        """
        super().__init__()
        self.shapenet_dir = shapenet_dir
        self.r2n2_dir = r2n2_dir
        self.views_rel_path = views_rel_path
        self.voxels_rel_path = voxels_rel_path
        self.load_textures = load_textures
        self.texture_resolution = texture_resolution
        self.return_feats = return_feats
        self.return_RTK = return_RTK
        # Examine if split is valid.
        if split not in ["train", "val", "test"]:
            raise ValueError("split has to be one of (train, val, test).")
        # Synset dictionary mapping synset offsets in R2N2 to corresponding labels.
        with open(
            os.path.join(SYNSET_DICT_DIR, "r2n2_synset_dict.json"), "r"
        ) as read_dict:
            self.synset_dict = json.load(read_dict)
        # Inverse dictionary mapping synset labels to corresponding offsets.
        self.synset_inv = {label: offset for offset, label in self.synset_dict.items()}

        # Store synset and model ids of objects mentioned in the splits_file.
        with open(splits_file) as splits:
            split_dict = json.load(splits)[split]

        self.return_images = True
        # Check if the folder containing R2N2 renderings is included in r2n2_dir.
        if not os.path.isdir(os.path.join(r2n2_dir, views_rel_path)):
            self.return_images = False
            msg = (
                "%s not found in %s. R2N2 renderings will "
                "be skipped when returning models."
            ) % (views_rel_path, r2n2_dir)
            warnings.warn(msg)

        self.return_voxels = return_voxels
        # Check if the folder containing voxel coordinates is included in r2n2_dir.
        if not os.path.isdir(os.path.join(r2n2_dir, voxels_rel_path)):
            self.return_voxels = False
            msg = (
                "%s not found in %s. Voxel coordinates will "
                "be skipped when returning models."
            ) % (voxels_rel_path, r2n2_dir)
            warnings.warn(msg)

        synset_set = set()
        # Store lists of views of each model in a list.
        self.views_per_model_list = []
        # Store tuples of synset label and total number of views in each category in a list.
        synset_num_instances = []
        for synset in split_dict.keys():
            # Examine if the given synset is present in the ShapeNetCore dataset
            # and is also part of the standard R2N2 dataset.
            if not (
                os.path.isdir(os.path.join(shapenet_dir, synset))
                and synset in self.synset_dict
            ):
                msg = (
                    "Synset category %s from the splits file is either not "
                    "present in %s or not part of the standard R2N2 dataset."
                ) % (synset, shapenet_dir)
                warnings.warn(msg)
                continue

            synset_set.add(synset)
            self.synset_start_idxs[synset] = len(self.synset_ids)
            # Start counting total number of views in the current category.
            synset_view_count = 0
            for model in split_dict[synset]:
                # Examine if the given model is present in the ShapeNetCore os.path.
                shapenet_path = os.path.join(shapenet_dir, synset, model)
                if not os.path.isdir(shapenet_path):
                    msg = "Model %s from category %s is not present in %s." % (
                        model,
                        synset,
                        shapenet_dir,
                    )
                    warnings.warn(msg)
                    continue
                self.synset_ids.append(synset)
                self.model_ids.append(model)

                model_views = split_dict[synset][model]
                # Randomly select a view index if return_all_views set to False.
                if not return_all_views:
                    rand_idx = torch.randint(len(model_views), (1,))
                    model_views = [model_views[rand_idx]]
                self.views_per_model_list.append(model_views)
                synset_view_count += len(model_views)
            synset_num_instances.append((self.synset_dict[synset], synset_view_count))
            model_count = len(self.synset_ids) - self.synset_start_idxs[synset]
            self.synset_num_models[synset] = model_count
        headers = ["category", "#instances"]
        synset_num_instances.append(("total", sum(n for _, n in synset_num_instances)))
        print(
            tabulate(synset_num_instances, headers, numalign="left", stralign="center")
        )

        # Examine if all the synsets in the standard R2N2 mapping are present.
        # Update self.synset_inv so that it only includes the loaded categories.
        synset_not_present = [
            self.synset_inv.pop(self.synset_dict[synset])
            for synset in self.synset_dict
            if synset not in synset_set
        ]

        if len(synset_not_present) > 0:
            msg = (
                "The following categories are included in R2N2's"
                "official mapping but not found in the dataset location %s: %s"
            ) % (shapenet_dir, ", ".join(synset_not_present))
            warnings.warn(msg)

    def __getitem__(self, model_idx, view_idxs=None):
        """
        Read a model by the given index.

        Args:
            model_idx: The idx of the model to be retrieved in the dataset.
            view_idx: List of indices of the view to be returned. Each index needs to be
                contained in the loaded split (always between 0 and 23, inclusive). If
                an invalid index is supplied, view_idx will be ignored and all the loaded
                views will be returned.

        Returns:
            dictionary with following keys:
            - verts: FloatTensor of shape (V, 3).
            - faces: faces.verts_idx, LongTensor of shape (F, 3).
            - synset_id (str): synset id.
            - model_id (str): model id.
            - label (str): synset label.
            - images: FloatTensor of shape (V, H, W, C), where V is number of views
                returned. Returns a batch of the renderings of the models from the R2N2 dataset.
            - R: Rotation matrix of shape (V, 3, 3), where V is number of views returned.
            - T: Translation matrix of shape (V, 3), where V is number of views returned.
            - K: Intrinsic matrix of shape (V, 4, 4), where V is number of views returned.
            - voxels: Voxels of shape (D, D, D), where D is the number of voxels along each
                dimension.
        """
        if isinstance(model_idx, tuple):
            model_idx, view_idxs = model_idx
        if view_idxs is not None:
            if isinstance(view_idxs, int):
                view_idxs = [view_idxs]
            if not isinstance(view_idxs, list) and not torch.is_tensor(view_idxs):
                raise TypeError(
                    "view_idxs is of type %s but it needs to be a list."
                    % type(view_idxs)
                )

        model_views = self.views_per_model_list[model_idx]
        if view_idxs is not None and any(
            idx not in self.views_per_model_list[model_idx] for idx in view_idxs
        ):
            msg = """At least one of the indices in view_idxs is not available.
                Specified view of the model needs to be contained in the
                loaded split. If return_all_views is set to False, only one
                random view is loaded. Try accessing the specified view(s)
                after loading the dataset with self.return_all_views set to True.
                Now returning all view(s) in the loaded dataset."""
            warnings.warn(msg)
        elif view_idxs is not None:
            model_views = view_idxs

        model = self._get_item_ids(model_idx)
        model_path = os.path.join(
            self.shapenet_dir, model["synset_id"], model["model_id"], "model.obj"
        )
        try:
            verts, faces, textures = self._load_mesh(model_path)
        except Exception:
            st()
        model["verts"] = verts
        model["faces"] = faces
        # model["textures"] = textures
        model["label"] = self.synset_dict[model["synset_id"]]

        model["images"] = None
        img_names, images, feats, Rs, Ts, voxel_RTs = [], [], [], [], [], []
        # Retrieve R2N2's renderings if required.
        if self.return_images:
            rendering_path = os.path.join(
                self.r2n2_dir,
                self.views_rel_path,
                model["synset_id"],
                model["model_id"],
                "rendering",
            )
            all_feats = torch.from_numpy(np.load(os.path.join(rendering_path, "feats.npy")))
            # Read metadata file to obtain params for calibration matrices.
            with open(os.path.join(rendering_path, "rendering_metadata.txt"), "r") as f:
                metadata_lines = f.readlines()
            for i, name in enumerate(model_views):
                # Read image.
                image_path = os.path.join(rendering_path, f"{name}.png")
                # image_path = os.path.join(rendering_path, "%02d.png" % i)
                raw_img = Image.open(image_path)
                if raw_img.size != (137, 137):
                    raw_img = raw_img.resize((137, 137))
                if np.array(raw_img).ndim != 3:
                    raw_img = cv2.cvtColor(np.array(raw_img), cv2.COLOR_GRAY2RGB)
                image = torch.from_numpy(np.array(raw_img) / 255.0)[..., :3]
                
                img_names.append(int(name))
                images.append(image.to(dtype=torch.float32))
                if self.return_feats:
                    feats.append(all_feats[i].to(dtype=torch.float32))

                if self.return_RTK:
                    # Get camera calibration.
                    azim, elev, yaw, dist_ratio, fov = [
                        float(v) for v in metadata_lines[i].strip().split(" ")
                    ]
                    dist = dist_ratio * MAX_CAMERA_DISTANCE
                    # Extrinsic matrix before transformation to PyTorch3D world space.
                    RT = compute_extrinsic_matrix(azim, elev, dist)
                    R, T = self._compute_camera_calibration(RT)
                    Rs.append(R)
                    Ts.append(T)
                    voxel_RTs.append(RT)

            # Intrinsic matrix extracted from the Blender with slight modification to work with
            # PyTorch3D world space. Taken from meshrcnn codebase:
            # https://github.com/facebookresearch/meshrcnn/blob/main/shapenet/utils/coords.py
            model["names"] = np.stack(img_names)
            model["images"] = torch.stack(images)
            if self.return_RTK:
                K = torch.tensor(
                    [
                        [2.1875, 0.0, 0.0, 0.0],
                        [0.0, 2.1875, 0.0, 0.0],
                        [0.0, 0.0, -1.002002, -0.2002002],
                        [0.0, 0.0, 1.0, 0.0],
                    ]
                )
                model["R"] = torch.stack(Rs)
                model["T"] = torch.stack(Ts)
                model["K"] = K.expand(len(model_views), 4, 4)
            if self.return_feats:
                model["feats"] = torch.stack(feats)

        voxels_list = []

        # Read voxels if required.
        voxel_path = os.path.join(
            self.r2n2_dir,
            self.voxels_rel_path,
            model["synset_id"],
            model["model_id"],
            "model.binvox",
        )
        if self.return_voxels:
            if not os.path.isfile(voxel_path):
                msg = "Voxel file not found for model %s from category %s."
                raise FileNotFoundError(msg % (model["model_id"], model["synset_id"]))

            with open(voxel_path, "rb") as f:
                # Read voxel coordinates as a tensor of shape (N, 3).
                voxel_coords = read_binvox_coords(f)
            # Align voxels to the same coordinate system as mesh verts.
            voxel_coords = align_bbox(voxel_coords, model["verts"])
            model["voxel_coords"] = voxel_coords
            voxels = utils_vox.voxelize_xyz(voxel_coords.unsqueeze(0),32,32,32).squeeze(0)
            # for RT in voxel_RTs:
            #     # Compute projection matrix.
            #     P = BLENDER_INTRINSIC.mm(RT)
            #     # Convert voxel coordinates of shape (N, 3) to voxels of shape (D, D, D).
            #     voxels = voxelize(voxel_coords, P, VOXEL_SIZE)
            #     voxels_list.append(voxels)
            model["voxels"] = voxels
        
        ## randomly select one view among all
        # num_views = model['images'].shape[0]
        # rand_view = random.randint(0,num_views-1)
        # model['images'] = model['images'][rand_view]
        # if self.return_RTK:
        #     model['R'] = model['R'][rand_view]
        #     model['T'] = model['T'][rand_view]
        #     model['K'] = model['K'][rand_view]
        # if self.return_feats:
        #     model["feats"] = model["feats"][rand_view]
        
        return model

    def _compute_camera_calibration(self, RT):
        """
        Helper function for calculating rotation and translation matrices from ShapeNet
        to camera transformation and ShapeNet to PyTorch3D transformation.

        Args:
            RT: Extrinsic matrix that performs ShapeNet world view to camera view
                transformation.

        Returns:
            R: Rotation matrix of shape (3, 3).
            T: Translation matrix of shape (3).
        """
        # Transform the mesh vertices from shapenet world to pytorch3d world.
        shapenet_to_pytorch3d = torch.tensor(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        RT = torch.transpose(RT, 0, 1).mm(shapenet_to_pytorch3d)  # (4, 4)
        # Extract rotation and translation matrices from RT.
        R = RT[:3, :3]
        T = RT[3, :3]
        return R, T

    def render(
        self,
        model_ids= None,
        categories= None,
        sample_nums= None,
        idxs= None,
        view_idxs= None,
        shader_type=HardPhongShader,
        device= "cpu",
        **kwargs
    ):
        """
        Render models with BlenderCamera by default to achieve the same orientations as the
        R2N2 renderings. Also accepts other types of cameras and any of the args that the
        render function in the ShapeNetBase class accepts.

        Args:
            view_idxs: each model will be rendered with the orientation(s) of the specified
                views. Only render by view_idxs if no camera or args for BlenderCamera is
                supplied.
            Accepts any of the args of the render function in ShapeNetBase:
            model_ids: List[str] of model_ids of models intended to be rendered.
            categories: List[str] of categories intended to be rendered. categories
                and sample_nums must be specified at the same time. categories can be given
                in the form of synset offsets or labels, or a combination of both.
            sample_nums: List[int] of number of models to be randomly sampled from
                each category. Could also contain one single integer, in which case it
                will be broadcasted for every category.
            idxs: List[int] of indices of models to be rendered in the dataset.
            shader_type: Shader to use for rendering. Examples include HardPhongShader
            (default), SoftPhongShader etc or any other type of valid Shader class.
            device: Device (as str or torch.device) on which the tensors should be located.
            **kwargs: Accepts any of the kwargs that the renderer supports and any of the
                args that BlenderCamera supports.

        Returns:
            Batch of rendered images of shape (N, H, W, 3).
        """
        idxs = self._handle_render_inputs(model_ids, categories, sample_nums, idxs)
        r = torch.cat([self[idxs[i], view_idxs]["R"] for i in range(len(idxs))])
        t = torch.cat([self[idxs[i], view_idxs]["T"] for i in range(len(idxs))])
        k = torch.cat([self[idxs[i], view_idxs]["K"] for i in range(len(idxs))])
        # Initialize default camera using R, T, K from kwargs or R, T, K of the specified views.
        blend_cameras = BlenderCamera(
            R=kwargs.get("R", r),
            T=kwargs.get("T", t),
            K=kwargs.get("K", k),
            device=device,
        )
        cameras = kwargs.get("cameras", blend_cameras).to(device)
        kwargs.pop("cameras", None)
        # pass down all the same inputs
        return super().render(
            idxs=idxs, shader_type=shader_type, device=device, cameras=cameras, **kwargs
        )
        
class R2N2(ShapeNetBase):
    def __init__(self, shapenet_dir, r2n2_dir, metadata_file, views_rel_path, voxels_rel_path):
        super().__init__()
        self.shapenet_dir = shapenet_dir
        self.r2n2_dir = r2n2_dir
        self.views_rel_path = views_rel_path
        self.voxels_rel_path = voxels_rel_path
        
        # Synset dictionary mapping synset offsets in R2N2 to corresponding labels.
        with open(os.path.join(SYNSET_DICT_DIR, "r2n2_synset_dict.json"), "r") as read_dict:
            self.synset_dict = json.load(read_dict)
        # Inverse dictionary mapping synset labels to corresponding offsets.
        self.synset_inv = {label: offset for offset, label in self.synset_dict.items()}
        
        # Store synset and model ids of objects mentioned in the splits_file.
        with open(metadata_file) as splits:
            metadata_dic = json.load(splits)
            
        synset_set = set()
        # Store lists of views of each model in a list.
        self.views_per_model_list = []
        # Store tuples of synset label and total number of views in each category in a list.
        synset_num_instances = []
        for synset in metadata_dic.keys():
            # Examine if the given synset is present in the ShapeNetCore dataset
            # and is also part of the standard R2N2 dataset.
            if not (
                os.path.isdir(os.path.join(shapenet_dir, synset))
                and synset in self.synset_dict
            ):
                msg = (
                    "Synset category %s from the splits file is either not "
                    "present in %s or not part of the standard R2N2 dataset."
                ) % (synset, shapenet_dir)
                warnings.warn(msg)
                continue

            synset_set.add(synset)
            self.synset_start_idxs[synset] = len(self.synset_ids)
            # Start counting total number of views in the current category.
            synset_view_count = 0
            for model in metadata_dic[synset]:
                # Examine if the given model is present in the ShapeNetCore os.path.
                shapenet_path = os.path.join(shapenet_dir, synset, model)
                if not os.path.isdir(shapenet_path):
                    msg = "Model %s from category %s is not present in %s." % (
                        model,
                        synset,
                        shapenet_dir,
                    )
                    warnings.warn(msg)
                    continue
                self.synset_ids.append(synset)
                self.model_ids.append(model)

                model_views = metadata_dic[synset][model]
                self.views_per_model_list.append(model_views)
                synset_view_count += len(model_views)
            synset_num_instances.append((self.synset_dict[synset], synset_view_count))
            model_count = len(self.synset_ids) - self.synset_start_idxs[synset]
            self.synset_num_models[synset] = model_count
        headers = ["category", "#instances"]
        synset_num_instances.append(("total", sum(n for _, n in synset_num_instances)))
        print(
            tabulate(synset_num_instances, headers, numalign="left", stralign="center")
        )
        
        
        print("Loading images...")
        name_dic = defaultdict(list)
        self.names, self.imgs, self.imgpaths = [], [], []
        imgsfolder = os.path.join(r2n2_dir, views_rel_path, "03001627")
        obj_imgs = glob(f"{imgsfolder}/*")
        obj_imgs = sorted(obj_imgs)
        for i, objf in enumerate(tqdm(obj_imgs, total=len(obj_imgs))):
            objname = os.path.basename(os.path.normpath(objf))
            
            for imgf in sorted(glob(f"{objf}/**/*.png")):
                imgnum = int(os.path.basename(imgf)[:-4])
                
                # img = cv2.imread(imgf)
                # if img.ndim != 3:
                #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                # if img.shape != (224,224,3):
                #     img = cv2.resize(img, (224,224))
                
                # self.imgs.append(torch.from_numpy(img).permute(2,0,1))
                self.imgpaths.append(imgf)
                self.names.append(int(i*15+imgnum))
                name_dic[objname].append(int(i*15+imgnum))
        
        print("Loading voxels...")
        self.voxels, self.voxelpaths = [], []
        self.voxelsfolder = os.path.join(r2n2_dir, voxels_rel_path, "03001627")
        obj_voxels = glob(f"{self.voxelsfolder}/*")
        obj_voxels = sorted(obj_voxels)
        for i, objf in enumerate(tqdm(obj_voxels, total=len(obj_voxels))):
            objname = os.path.basename(os.path.normpath(objf))
            
            # modelpath = os.path.join(shapenet_dir, "03001627", objname, "model.obj")
            # try:
            #     verts, faces, textures = self._load_mesh(modelpath)
            # except Exception:
            #     st()
                
            # voxf = os.path.join(objf, "model.binvox")
            # with open(voxf, "rb") as f:
            #     voxel_coords = read_binvox_coords(f)
            # voxel_coords = align_bbox(voxel_coords, verts)
            # voxel = utils_vox.voxelize_xyz(voxel_coords.unsqueeze(0),32,32,32).squeeze(0,1)
            
            for _ in range(len(name_dic[objname])):
                # self.voxels.append(voxel)
                self.voxelpaths.append(objf)
        
        # self.name_dic = {label: offset  for objname, numlist in name_dic.items()}
        self.name_dic = {}
        for objname, numlist in name_dic.items():
            for num in numlist:
                self.name_dic[num] = objname
        
        assert len(self.imgs) == len(self.voxels)
    
    def __len__(self):
        return len(self.imgpaths)
    
    def __getitem__(self, idx):
        
        img = cv2.imread(self.imgpaths[idx])
        if img.ndim != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.shape != (224,224,3):
            img = cv2.resize(img, (224,224))
        img = torch.from_numpy(img).permute(2,0,1)
        
        objname = os.path.basename(os.path.normpath(self.voxelpaths[idx]))
        modelpath = os.path.join(self.shapenet_dir, "03001627", objname, "model.obj")
        try:
            verts, faces, textures = self._load_mesh(modelpath)
        except Exception:
            st()
        
        voxf = os.path.join(self.voxelpaths[idx], "model.binvox")
        with open(voxf, "rb") as f:
            voxel_coords = read_binvox_coords(f)
        voxel_coords = align_bbox(voxel_coords, verts)
        voxel = utils_vox.voxelize_xyz(voxel_coords.unsqueeze(0),32,32,32).squeeze(0,1)
        
        # gt_textures = torch.ones_like(verts)*torch.tensor([[0.0, 1.0, 0.0]])
        # gt_mesh = Meshes(
        #     verts=verts.unsqueeze(0),
        #     faces=faces.unsqueeze(0),
        #     textures=TexturesVertex(gt_textures.unsqueeze(0))
        # )
        # scene = plot_scene({
        #     "Figure1": {
        #         "GT Mesh": gt_mesh
        #     }
        # })
        # scene.show()
        
        dic = {
            'names': self.names[idx],
            'images': img,
            'voxels': voxel
        }
        # dic = {
        #     'names': self.names[idx],
        #     'images': self.imgs[idx],
        #     'voxels': self.voxels[idx]
        # }
        return dic
