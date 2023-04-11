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
|-- Bench
|-- Chair
|   |-- agam
|   |    -- step_0.obj
|   |    -- step_1.obj
|   |    -- step_2.obj
|   |    -- step_3.obj
|   |    -- step_4.obj
|   |    -- step_5.obj
|   |-- applaro
|   ...
| ...
```

The above structure is purely for easy reference. For training, the script will save numbered step-wise .OBJ models under a `obj_models/` directory within your dataset directory. It will also save numbered step-wise .OFF models under a `off_models/` directory within your dataset directory. ie.
```
<dataset dir>
|-- obj_models
|   |-- 00000.obj
|   |-- 00001.obj
|   ...
|-- off_models
|   |-- 00000.off
|   |-- 00001.off
|   ...
```