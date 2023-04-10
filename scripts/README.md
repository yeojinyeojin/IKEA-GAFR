### Getting CAD Model Ground Truths
Run the following from this directory:
```python
python cad_model.py --dataset_path <PATH TO IKEA-MANUAL DATASET> --output_dir <DIRECTORY TO WRITE CAD MODELS TO>
```
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
|   |-- appalro
|   ...
| ...
```