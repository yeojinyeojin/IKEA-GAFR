### Getting CAD Model Ground Truths
Run the following from this directory:
```python
python cad_model.py --args 
```

Arguments:
- `--dataset_path`: Path to IKEA-Manual Dataset
- `--output_dir`: Directory to save CAD Models
- `--save_obj_files`: Include if you want to save models in .OBJ format
- `--save_off_files`: Include if you want to save models in .OFF format (.OFF models used for mesh voxelization)
- `--visualize`: Include if you want to visualize each sample

The script will create the output directory if it does not exist and writes step-wise CAD models following the same file structure as the IKEA-Manual dataset ie.
```
<output_dir>
|-- Bench/
|-- Chair/
|   |-- agam/
|   |    -- step_0.obj
|   |    -- step_1.obj
|   |    -- step_2.obj
|   |    -- step_3.obj
|   |    -- step_4.obj
|   |    -- step_5.obj
|   |-- applaro/
|   ...
| ...
```

The above structure is purely for easy reference. For training, the script will save numbered step-wise .OBJ models under a `models/` directory within your dataset directory. It will also save numbered step-wise .OFF models under a `off_models/` directory within your dataset directory.
```
<dataset dir>
|-- models/
|   |-- 00000.obj
|   |-- 00001.obj
|   ...
|-- off_models/
|   |-- 00000.off
|   |-- 00001.off
|   ...
```
### Verifying Ground Truth Quality
Download zipped data from [here](https://drive.google.com/file/d/1xsCA_YD8UuZhmxxsdNxXkASQ6GL0s_dF/view?usp=share_link). Unzip and move all directories into your main dataset directory. Dataset directory should have the following structure: 
 ```
<dataset dir>
|-- code/
|-- gt_voxels_32_x_32/
|   |-- 00000.obj
|   |-- 00000.off
|   |-- 00001.obj
|   |-- 00001.off
|   ... 
|-- line_seg/
|-- mask/
|-- models/
|   |-- 00000.obj
|   |-- 00001.obj
|   ...
|-- off_models/
|   |-- 00000.off
|   |-- 00001.off
|   ...
|-- off_models_32_x_32/
|   |-- 00000.obj
|   |-- 00000.off
|   |-- 00001.obj
|   |-- 00001.off
|   ...
|-- parts/
|-- pdfs/
|-- main_data.json
|-- README.md

```

For verifying ground truth quality, run the following command from this directory:
```python
python cad_model.py --dataset_path <path to dataset> --visualize --start_idx <model idx to start visualizing>
```
This will open up a visualizer in your browser for each model in range #start_idx to #start_idx + 130 and will pause the script until you have manually verified ground truth quality, closed the window, and hit enter to visualize the next model. GT Voxels will be in green, GT Meshes will be in red.