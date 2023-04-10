## Scripting

Poppler installation steps: [github](https://github.com/Belval/pdf2image)

IKEA-Manual data can be downloaded [here](https://download.cs.stanford.edu/viscam/ikea_manual/dataset.zip)

Dataset folder structure:
```
<root_dir>
|-- dataset
|-- ikea_man
|   |--code
|   |--line_seg
|   |--mask
|   |--parts
|   |__pdfs
|
|-- scripts
|__ README.md
```


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