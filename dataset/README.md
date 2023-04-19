# Dataset metadata

Dataset600 (filtered version of the full ikea_man dataset with 600 annotations instead of 953) is [here](https://drive.google.com/file/d/1BzTZpGj_EgfrdsDYBFvLb0nBfBpTz_8C/view?usp=share_link)

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