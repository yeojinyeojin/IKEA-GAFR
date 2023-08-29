# IKEA-GAFR

We propose IKEA-GAFR, a pipeline that performs key-step recognition and single-view sketch-based 3D reconstruction directly from IKEA furniture assembly manuals. We first identified key steps from IKEA manuals by formulating the task as an object detection problem with 2 classes (relevant & irrelevant). We trained a YOLOv5 network on the IKEA-Manual dataset with bounding box and relevance labels annotations for key step recognition. For single-view sketch-based 3D reconstruction, we proposed a part-segmented point cloud reconstruction model based on SketchSampler, by introducing part-wise density maps, which encode the probability of 3D points belonging to different parts projecting to each 2D location. In order to incorporate both reconstruction and part segmentation into the model’s learning paradigm, we used a combined loss function of Chamfer distance for point cloud reconstruction, MAE loss for density map prediction, point-wise softmax cross-entropy loss for implicit part-wise segmented density map prediction, and location-aware segmentation loss for explicit segmentation supervision on the final point cloud.

### Dependencies
Poppler installation steps: [github](https://github.com/Belval/pdf2image)

### Repository Structure
`pointcloud` (default) branch contains codes for joint 3D point cloud reconstruction and part segmentation network based on SketchSampler. `voxel` branch contains codes for 3D voxel reconstruction based on Pix2Vox.

## Dataset

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


