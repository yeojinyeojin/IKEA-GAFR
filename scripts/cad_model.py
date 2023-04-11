import json
import os
import torch
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj, save_obj
from pytorch3d.transforms import Rotate, Translate, matrix_to_euler_angles
from pytorch3d.renderer import TexturesVertex, look_at_view_transform
from pytorch3d.renderer.cameras import FoVPerspectiveCameras
from pytorch3d.vis.plotly_vis import plot_scene
import argparse as ap

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

def main(args):
    dataset_path = args.dataset_path
    data_json = open(os.path.join(dataset_path, "main_data.json"))
    data = json.load(data_json)

    counter = 0
    for sample_num, sample in enumerate(data):
        category = sample['category']
        name = sample['name']
        print(f'{category}: {name}')
        model_dir = os.path.join(dataset_path, "parts", category, name)
        part_names = sorted(os.listdir(model_dir))
        # prev_part_transforms = [None]*len(part_names)
        
        for step_num, step_data in enumerate(sample['steps']):
            print(f'Step {step_num}')
            # extrinsics = step_data['extrinsics']
            # intrinsics = step_data['intrinsics']
            model_verts = None
            model_faces = None
            
            step_part_nums, step_part_idxs = preprocess_part_idxs(step_data['parts'])
            # print(f'Step {step_num}')
            # print(f'Part Nums {step_part_nums}')
            # print(f'Part Idxs {step_part_idxs}')
            for part_idx, part_num in zip(step_part_idxs, step_part_nums):
                part_num = int(part_num)
                part_path = os.path.join(model_dir, part_names[part_num]) # we want to index into part_names with the model_number as that will be the correct index
                # part_extrinsics = torch.tensor(extrinsics[part_idx]) # we want to index into extrinsics in the same order the parts are presented
                # print((matrix_to_euler_angles(part_extrinsics[:-1, :-1].unsqueeze(0), convention="XYZ")))
                # print(torch.rad2deg(matrix_to_euler_angles(part_extrinsics[:-1, :-1].unsqueeze(0), convention="XYZ")))
                # continue
                # part_R = Rotate(part_extrinsics[:-1, :-1]) #Note coordinate system and how transformations are applied, check under Transform3D class https://pytorch3d.readthedocs.io/en/latest/modules/transforms.html
                # part_t = Translate(part_extrinsics[:-1, -1].reshape(-1, 3)) #Note coordinate system and how transformations are applied, check under Transform3D class https://pytorch3d.readthedocs.io/en/latest/modules/transforms.html
                
                # part_t = Translate(torch.fliplr((part_extrinsics[:-1, -1]/100.0).reshape(-1, 3))) #Divide by 100 (unit issue) and flip (coordinate system issue)???????                
                # part_t = Translate(((part_extrinsics[:-1, -1]/100.0).reshape(-1, 3)))

                # if step_num == 0:
                #     part_Rt = part_R.compose(part_t) #rotate then translate
                # else:
                #     if prev_part_transforms[part_num] is not None:
                #         part_Rt = prev_part_transforms[part_num].compose(part_R).compose(part_t) #compose with previous transform
                
                # prev_part_transforms[part_num] = part_Rt
                part_verts, part_face_data, _ = load_obj(part_path)
                # part_verts = part_Rt.transform_points(part_verts)
                part_faces = part_face_data.verts_idx
                
                if model_verts is None:
                    model_verts = part_verts
                    model_faces = part_faces
                else:
                    part_faces += len(model_verts)
                    model_verts = torch.cat([model_verts, part_verts], dim = 0)
                    model_faces = torch.cat([model_faces, part_faces], dim = 0)
            # return

            # model_textures = torch.ones_like(model_verts)*torch.tensor([0.7, 0.7, 1.0])
            # model_mesh = Meshes(
            #     verts=model_verts.unsqueeze(0),
            #     faces=model_faces.unsqueeze(0),
            #     textures=TexturesVertex(model_textures.unsqueeze(0))
            # )
            model_write_dir = os.path.join(args.output_dir, category, name + "/")
            print(f'Writing model to {model_write_dir}')
            if not os.path.exists(model_write_dir):
                os.makedirs(model_write_dir)
            save_obj(f=os.path.join(model_write_dir, "step_" + str(step_num) + ".obj"), verts=model_verts, faces=model_faces)
            
            model_write_dir = os.path.join(os.path.join(args.dataset_path), "models" + "/")
            if not os.path.exists(model_write_dir):
                os.makedirs(model_write_dir)
            save_obj(f=os.path.join(model_write_dir, "{:05d}".format(counter) + '.obj'), verts=model_verts, faces=model_faces)
            counter += 1
            
        #     cam_R, cam_T = look_at_view_transform(
        #         dist=3,
        #         elev=0,
        #         azim=torch.linspace(-180.0, 180.0, 10),
        #     )

        #     cameras = FoVPerspectiveCameras(R=cam_R, T=cam_T)

        #     scene = plot_scene({
        #         "Figure1": {
        #             "Meshes": model_mesh
        #         }
        #     })
        #     scene.show()
        #     break
        # break
    print("Done getting all CAD models")

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, help='Path to IKEA-Manual Dataset')
    parser.add_argument('--output_dir', type=str, help="Directory to write CAD models to", default="./cad_models")
    args = parser.parse_args()
    main(args)
