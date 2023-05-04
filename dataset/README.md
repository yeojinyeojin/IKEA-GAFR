# Dataset metadata

- `Dataset600`: filtered version of the full ikea_man dataset with 600 annotations instead of 953 is [here](https://drive.google.com/file/d/1BzTZpGj_EgfrdsDYBFvLb0nBfBpTz_8C/view?usp=share_link)
- `ShapenetRotate`: Rotated version of `Shapenet` dataset with 73024 images is [here](https://drive.google.com/file/d/1DBe7K1p8v07HTQ_Jfg_lAVsWTmuzLBlO/view?usp=share_link)
- `ShapenetRotateSketch`: Sketched version of `ShapenetRotate' dataset with 73024 images is [here](https://drive.google.com/file/d/1jQra9LeUnitPCmuNyV22esluGTzek2LQ/view?usp=share_link)
- `R2N2_sketch`: Sketched version of `R2N2' ShapenetCore subset dataset with 67770 images is [here](https://drive.google.com/file/d/1kXH2nw5v8fsBK8bvmI3N_xFChjVMiOUO/view?usp=sharing)


# Data Augmentation Using R2N2_Shapenet_Dataset

1. Download `r2n2_shapenet_dataset` from [here](https://drive.google.com/file/d/1VoSmRA9KIwaH56iluUuBEBwCbbq3x7Xt/view?usp=sharing) and put it under `dataset` folder.
2. Convert images to line drawings
    ```
    cd scripts
    python img2line.py --img_dir [path to r2n2 dataset] --out_dir [path to output directory]
    ```
    **NOTE**: If you want to rotate `shapenet` objects in random angles and then generate line drawings:
    ```
    python img2line.py --rotate --vox_dir [path to shapenet dataset] --out_dir [path to output directory]
    ```
3. Organize output images into IKEAManual format
    ```
    python organize_lineimgs.py --img_dir [path to line drawings] --orig_dir [path to R2N2 ShapeNetRendering folder] --out_dir [path to output directory]
    ```
    **NOTE**: Put output directory as `r2n2_shapenet_dataset/r2n2/LineDrawings`.
    **NOTE**: You need to generate line drawings of original images and rotated images separately.

Expected Dataset folder structure:
```
<root_dir>
|-- dataset
|   |__ r2n2_shapenet_dataset
|       |-- r2n2
|       |   |-- ShapeNetRendering/03001627
|       |   |-- ShapeNetVoxels/03001627
|       |   |__ LineDrawings/03001627
|       |-- shapenet/03001627
|       |__ line_split.json
|-- scripts
|-- ...
|__ README.md
```
**NOTE**: `03001627` is the synid, which should be included for `orig_dir` and `out_dir` when running `organize_lineimgs.py`.

## Dataset `main_data.json` File Structure
`main_data.json` contains annotations for each objects in the following format:
```
{
    // Object metadata.
    'category': 'Bench',
    'name': 'applaro',
    
     // Annotations for each step.
    'steps': [...],
    
     // Connection relation between primitive assembly parts.
    'connection_relation': [...], 
    
     // Geometric equivalence relation between primitive assembly parts.
    'geometric_equivalence_relation': [...],
    
     // Tree-structured assembly plan in list format.
    'assembly_tree': [...],
    
     // Number of primitive assembly parts.
    'parts_ct': 4, 
    }
```

The `steps` field contains a list of step annotation in the following format:

```
{
    // List of involved assembly parts in this step.
    'parts': ['0,1,2', '3']
    
    // List of pairwise connection relationships.
    'connections': [['0,1,2', '3']]
    
    // The page where this step is in.
    'page_id': 4,
    
    // The step id.
    'step_id': 1,
    
    // A list of masks for each assembly parts.
    // Use pycocotools.masks.decode to decode each mask
    // into an numpy array.
    'masks': masks,
    
    // A list of intrinsic matrices for each assembly part.
    'intrinsics': [...],
    
    // A list of extrinsic matrices for each assembly part.
    'extrinsics': [...],
    
    // The split for the segmentation task.
    'part_segmentation_split': 'test',
    
    // The index of the step in the whole dataset.
    'step_id_global': 10,
}
```

## Augmentation and Pretraining on R2N2_Shapenet_Dataset

1. Download, augment, and format the dataset by following instructions under `dataset`'s `README.md`

2. Pretrain with R2N2 dataset:
```
python train_3drecon.py --r2n2 --r2n2_dir [path to r2n2_shapenet_dataset]
```

3. Finetune with IKEA-Manual dataset:
```
python train_3drecon.py --load_checkpoint [path to model checkpoint from previous pretraining] --dataset_path [path to folder that contains "images" and "labels" and "off_models_32_x_32"]
```
**NOTE**: add `--chairs_only` flag if you only want to finetune on chairs samples

4. Run inference:
```
python test_3dcon.py --dataset_path [path to folder that contains "images" and "labels" and "off_models_32_x_32"] --load_checkpoint [path to model checkpoint from finetuning]
```