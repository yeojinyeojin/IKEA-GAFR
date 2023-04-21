import json
import os
import torch
import argparse as ap
import shutil
import numpy as np
import cv2

from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj, save_obj
from pytorch3d.transforms import Rotate, Translate, matrix_to_euler_angles
from pytorch3d.renderer import TexturesVertex, look_at_view_transform
from pytorch3d.renderer.cameras import FoVPerspectiveCameras
from pytorch3d.vis.plotly_vis import plot_scene

# p3d_coordinate_frame = Rotate(torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
# opencv_coordinate_frame = Rotate(torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]))
# opengl_coordinate_frame = Rotate(torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]))

def preprocess_part_idxs(part_list):
    '''
    Helper function that takes in list of parts from json and splits it into individual part numbers and 
    indices into extrinsics/intrinsics data
    '''
    step_part_nums = []
    step_part_idxs = []

    for i, part_num in enumerate(part_list):
        if len(part_num) > 1:
            split = part_num.split(",")
            step_part_nums.extend(split)
            step_part_idxs.extend([i]*len(split))
        else:
            step_part_nums.append(part_num)
            step_part_idxs.append(i)
    
    return step_part_nums, step_part_idxs

def obj_to_off(verts, faces, dataset_dir, model_num):
    print("Saving model {:05d} as OFF file".format(model_num))
    numbers_data = torch.tensor([[len(verts), len(faces), 0]], dtype=int).cpu().numpy()
    vert_data = verts.cpu().numpy()
    face_data = torch.cat([torch.ones((faces.shape[0], 1))*faces.shape[1], faces], dim = 1).to(int).cpu().numpy()
    tmp_dir = "./tmp"
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    if not os.listdir(tmp_dir):
        print("HERE")
        with open(os.path.join(tmp_dir, "off.txt"), 'w') as f:
            f.write("OFF")
        f.close()

    np.savetxt(os.path.join(tmp_dir, "numbers.txt"), numbers_data, fmt="%d")
    np.savetxt(os.path.join(tmp_dir, "verts.txt"), vert_data)
    np.savetxt(os.path.join(tmp_dir, "faces.txt"), face_data, fmt="%d")
    output_dir = os.path.join(dataset_dir, "gt_off_unscaled")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    off_path = os.path.join(tmp_dir, "off.txt")
    numbers_path = os.path.join(tmp_dir, "numbers.txt")
    verts_path = os.path.join(tmp_dir, "verts.txt")
    faces_path = os.path.join(tmp_dir, "faces.txt")

    with open(os.path.join(output_dir, "{:05d}.off".format(model_num)),'wb') as out_file:
        for f_num, f in enumerate([off_path, numbers_path, verts_path, faces_path]):
            with open(f,'rb') as fd:
                shutil.copyfileobj(fd, out_file)
                if f_num == 0:
                    out_file.write(b"\n")

def off_to_obj(off_filepath, return_mesh_object=False):
    output_path = os.path.join(os.path.dirname(off_filepath), os.path.basename(off_filepath).split(".")[0] + ".obj")
    with open(off_filepath, 'r') as f:
        data = f.read().split("\n")
        num_verts = int(data[1].split(" ")[0])
        num_faces = int(data[1].split(" ")[1])
        vert_start_idx = 2
        face_start_idx = vert_start_idx + num_verts
        verts = np.array([v.split(" ") for v in data[vert_start_idx: face_start_idx]]).astype(np.float64)
        verts = torch.from_numpy(verts)
        faces = np.array([f.split(" ")[1:] for f in data[face_start_idx:] if len(f) > 0]).astype(int)
        faces = torch.from_numpy(faces)
        
    f.close()
    print(f'Saving model to {output_path}...')
    save_obj(f=output_path, verts=verts, faces=faces)
    
    if return_mesh_object:
        textures = torch.zeros_like(verts) + torch.tensor([[0.7, 0.7, 1.0]])
        mesh = Meshes(
            verts=verts.unsqueeze(0),
            faces=faces.unsqueeze(0),
            textures=TexturesVertex(textures.unsqueeze(0))
        )

        return mesh

def main(args):
    dataset_path = args.dataset_path
    data_json = open(os.path.join(dataset_path, "main_data.json"))
    data = json.load(data_json)

    counter = -1
    for sample in data:
        category = sample['category']
        name = sample['name']
        # print(f'Processing {category}: {name}...')
        model_dir = os.path.join(dataset_path, "parts", category, name)
        part_names = sorted(os.listdir(model_dir))
        if args.prediction_frame == "object-centric":
            prev_part_transforms = [None]*len(part_names)
        
        for step_num, step_data in enumerate(sample['steps']):
            counter += 1
            #if counter < args.start_idx:
            #    continue
            #elif counter > (args.start_idx + 130):
            #    return

            #if args.end_idx:
            #    if counter > args.end_idx:
            #        return
            print(f'Model # {counter}')
            model_verts = None
            model_faces = None
            
            step_part_nums, step_part_idxs = preprocess_part_idxs(step_data['parts'])
            extrinsics = step_data['extrinsics']
            intrinsics = step_data['intrinsics']

            for part_idx, part_num in zip(step_part_idxs, step_part_nums):
                part_num = int(part_num)
                part_path = os.path.join(model_dir, part_names[part_num]) # we want to index into part_names with the model_number as that will be the correct index
                part_verts, part_face_data, _ = load_obj(part_path)
                part_faces = part_face_data.verts_idx
                if args.prediction_frame == "object-centric":
                    part_extrinsics = torch.tensor(extrinsics[part_idx]) # we want to index into extrinsics in the same order the parts are presented
                    part_R = Rotate(part_extrinsics[:-1, :-1]) #Note coordinate system and how transformations are applied, check under Transform3D class https://pytorch3d.readthedocs.io/en/latest/modules/transforms.html
                    part_t = Translate(torch.fliplr((part_extrinsics[:-1, -1]/100.0).reshape(-1, 3))) #Divide by 100 (unit issue) and flip (coordinate system issue)???????                

                    if step_num == 0:
                        part_Rt = part_R.compose(part_t) #rotate then translate
                    else:
                        if prev_part_transforms[part_num] is not None:
                            part_Rt = prev_part_transforms[part_num].compose(part_R).compose(part_t) #compose with previous transform
                    
                    prev_part_transforms[part_num] = part_Rt
                    part_verts = part_Rt.transform_points(part_verts)
                
                if model_verts is None:
                    model_verts = part_verts
                    model_faces = part_faces
                else:
                    part_faces += len(model_verts)
                    model_verts = torch.cat([model_verts, part_verts], dim = 0)
                    model_faces = torch.cat([model_faces, part_faces], dim = 0)

            if args.save_obj_files:

                #Saving in output directory using original file structure by category and model name
                model_write_dir = os.path.join(args.output_dir, category, name + "/")
                print(f'Writing model to {model_write_dir}')
                if not os.path.exists(model_write_dir):
                    os.makedirs(model_write_dir)
                save_obj(f=os.path.join(model_write_dir, "step_" + str(step_num) + ".obj"), verts=model_verts, faces=model_faces)


                #Saving all in one directory "models" using counter as save index
                model_write_dir = os.path.join(os.path.join(args.dataset_path), "models" + "/")
                if not os.path.exists(model_write_dir):
                    os.makedirs(model_write_dir)
                save_obj(f=os.path.join(model_write_dir, "{:05d}".format(counter) + '.obj'), verts=model_verts, faces=model_faces)
            
            if args.save_off_files:
                obj_to_off(model_verts, model_faces, dataset_path, counter)
        
            if args.visualize:
                try:
                    scaled_model_verts, scaled_model_face_data, _ = load_obj(os.path.join(dataset_path, "gt_off_32_x_32", "{:05d}.obj".format(counter)))
                    scaled_model_verts += torch.tensor([[0.0, 32.0, 0.0]])
                    scaled_model_textures = torch.ones_like(scaled_model_verts)*torch.tensor([[1.0, 0.0, 0.0]])
                    scaled_model_mesh = Meshes(
                        verts=scaled_model_verts.unsqueeze(0),
                        faces=scaled_model_face_data.verts_idx.unsqueeze(0),
                        textures=TexturesVertex(scaled_model_textures.unsqueeze(0))
                    )

                except:
                    scaled_model_mesh = off_to_obj(os.path.join(dataset_path, "gt_off_32_x_32", "{:05d}.off".format(counter)), return_mesh_object=True)

                try:
                    gt_vox_verts, gt_vox_face_data, _ = load_obj(os.path.join(dataset_path, "gt_voxels_32_x_32", "{:05d}.obj".format(counter)))
                    gt_textures = torch.ones_like(gt_vox_verts)*torch.tensor([[0.0, 1.0, 0.0]])
                    gt_mesh = Meshes(
                        verts=gt_vox_verts.unsqueeze(0),
                        faces=gt_vox_face_data.verts_idx.unsqueeze(0),
                        textures=TexturesVertex(gt_textures.unsqueeze(0))
                    )
                
                except:
                    gt_mesh = off_to_obj(os.path.join(dataset_path, "gt_voxels_32_x_32", "{:05d}.off".format(counter), return_mesh_object=True)

                scene = plot_scene({
                    "Figure": {
                        "Mesh Mesh": scaled_model_mesh,
                        "Voxel Mesh": gt_mesh
                    }
                })
                scene.show()
                input("Close viz window and press enter to continue to next sample...")
            

    if args.save_off_files:
        shutil.rmtree("./tmp")
    print("Done getting all CAD models")

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, help='Path to IKEA-Manual Dataset')
    parser.add_argument('--output_dir', type=str, help="Directory to write CAD models to", default="./cad_models")
    parser.add_argument('--prediction_frame', type=str, help="Either 'canonical' or 'object-centric'", default="canonical")
    parser.add_argument('--save_off_files', action=ap.BooleanOptionalAction)
    parser.add_argument('--save_obj_files', action=ap.BooleanOptionalAction)
    parser.add_argument('--visualize', action=ap.BooleanOptionalAction)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=None)
    args = parser.parse_args()
    main(args)
