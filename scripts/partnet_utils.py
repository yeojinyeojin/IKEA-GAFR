from tqdm import tqdm
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes, mesh_face_areas_normals
import os
import torch
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRendererWithFragments,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    TexturesAtlas,
)

from pytorch3d.renderer.mesh.shader import SoftDepthShader, HardDepthShader
import cv2
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import argparse as ap
from skimage import io, morphology, img_as_bool, segmentation
from scipy import ndimage as ndi
import shutil

device = "cuda:1" if torch.cuda.is_available() else "cpu"

def get_mesh_renderer(args):
    R, T = look_at_view_transform(
        dist=10,
        elev=-8,
        azim=torch.linspace(-180.0, 180.0, args.n_views)
    )
    R = R.to(device)
    T = T.to(device)

    fx, fy =-args.f, -args.f
    cx, cy = args.imsize//2, args.imsize//2
    cameras = PerspectiveCameras(focal_length = torch.tensor([[fx, fy]]).to(device), principal_point=torch.tensor([[cx, cy]]).to(device), in_ndc=False, R=R, T=T, image_size=torch.tensor([[args.imsize, args.imsize]]).to(device))
    raster_settings = RasterizationSettings(
        image_size=args.imsize,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,

    )

    lights = PointLights(device=device, location=[[0.75, 0.75, -0.75]]).to(device)

    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    ).to(device)

    return renderer, R, T

def create_if_empty(path):
    if not os.path.exists(path + "/"):
        os.makedirs(path)
    return

def get_normed_depth_from_zbuf(zbuf, unnormed=False):
    if len(zbuf.unique()) == 1:
        return None
    min_val = torch.sort(zbuf.unique())[0][1]

    if unnormed:
        depth = zbuf.cpu().numpy()
        #print(depth.max())
        return depth
    
    zbuf[torch.where(zbuf == zbuf.min())] = min_val
    zbuf = (zbuf - zbuf.min())/(zbuf.max() - zbuf.min())
    zbuf[torch.where(zbuf != torch.tensor(0.0))] = 1.0 - zbuf[torch.where(zbuf != 0.0)]
    depth = zbuf.cpu().numpy()

    return depth

def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    return edged

def get_edgemap(im):
    #img = np.uint8(im*255.0)
    img = im
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, ksize=(3,3), sigmaX=cv2.BORDER_DEFAULT)
    canny = auto_canny(img_blur, sigma=0.9)

    return canny

def get_canny(args, renderer, model_verts, model_faces, num_verts, num_faces, view_idx=0):
    curr_v_idx = 0
    curr_f_idx = 0
    model_canny = np.zeros((args.imsize, args.imsize, len(num_verts)))
    model_depth = np.ones((args.imsize, args.imsize, len(num_verts)))*-1.0
    for i in range(len(num_verts)):
        nv = num_verts[i]
        nf = num_faces[i]
        part_verts = model_verts[curr_v_idx: curr_v_idx + nv]
        part_faces = model_faces[curr_f_idx: curr_f_idx + nf] - curr_v_idx

        part_mesh = Meshes(
            verts=part_verts.unsqueeze(0),
            faces=part_faces.unsqueeze(0),
            textures=TexturesVertex((torch.zeros_like(part_verts) + torch.tensor([0.7, 0.7, 1.0])).unsqueeze(0))
        ).to(device)

        part_images, part_fragments = renderer(part_mesh.extend(args.n_views))
        part_image = np.uint8((part_images[view_idx, :, :, :-1]*255.0).cpu().numpy())
        part_canny = get_edgemap(part_image)
        model_canny[:, :, i] = part_canny

        part_depth_values = part_fragments.zbuf.to(device)
        part_depth = get_normed_depth_from_zbuf(part_depth_values[view_idx, :, :, 0], unnormed=True)
        depth = get_normed_depth_from_zbuf(part_depth_values[view_idx, :, :, 0])
        if depth is None:
            curr_v_idx += nv
            curr_f_idx += nf
            continue
        else:
            model_depth[:, :, i] = part_depth

        curr_v_idx += nv
        curr_f_idx += nf

    final_canny = np.zeros((args.imsize, args.imsize))
    model_depth[np.where(model_depth == -1)] = 100000
    closest_points = np.argmin(model_depth, axis=2)
    num_nonzero = 0

    for y in range(args.imsize):
        for x in range(args.imsize):
            if not (model_canny[y, x] == 0).all():
                idxs = np.argwhere(model_depth[y, x] == np.amin(model_depth[y, x])).reshape(-1)
                for idx in idxs:
                    idx = idx.squeeze()
                    if model_canny[y, x, idx] != 0:
                        final_canny[y, x] = model_canny[y, x, idx]
    
    return np.uint8(np.dstack([final_canny, final_canny, final_canny]))

def get_densities(args, model_verts, model_faces, num_verts, num_faces, area_probs, R, T, K):
    curr_v_idx = 0 #Maintain idx into model_verts to correctly index part vertices and faces
    curr_f_idx = 0 #Same as above

    total_points = 0 #Maintain counter for total number of points in model point cloud to ensure final sampled point cloud has args.n_samples 
    seg_density_map = torch.zeros((args.imsize, args.imsize, args.n_classes)) #Initialize segmentation density map (H x W x args.n_classes)
    density_map = torch.zeros((args.imsize, args.imsize)) #Initialize density map (H x W)
    canvas = torch.zeros((args.imsize, args.imsize, 3))
    model_points = None

    for part_idx in range(len(num_verts)): #Iterate through each part
        part_verts = model_verts[curr_v_idx: curr_v_idx + num_verts[part_idx]] #Get part vertices
        part_faces = model_faces[curr_f_idx: curr_f_idx + num_faces[part_idx]] - curr_v_idx #Get part faces and adjust to get correct indices into part_verts
        part_textures = torch.zeros_like(part_verts) + torch.tensor([0.7, 0.7, 1.0]) #Initialize standard texture to create mesh

        curr_v_idx += len(part_verts) #Increment vert counter
        curr_f_idx += len(part_faces) #Increment face counter

        #Create mesh for part
        part_mesh = Meshes(
            verts=part_verts.unsqueeze(0),
            faces=part_faces.unsqueeze(0),
            textures=TexturesVertex(part_textures.unsqueeze(0))
        )

        if part_idx == len(num_verts) - 1: #If we are at the last part, just sample however many remaining points are necessary to get args.n_samples
            num_samples = args.n_samples - total_points #Will still be roughly proportional to part area
        else:
            num_samples = max(1, int(area_probs[part_idx]*args.n_samples)) #Get number of points to sample from part 

        total_points += num_samples #Increment total points counter

        try: 
            part_points = sample_points_from_meshes(part_mesh, num_samples=num_samples) #Sample point cloud from part mesh
        except:
            raise Exception

        part_points = part_points.to(device).squeeze(0) #Send points to GPU
        if model_points is None:
            model_points = torch.cat([part_points, torch.ones((len(part_points), 1), device=part_points.device)*part_idx], dim = 1) #labeled pcd ie. first 3 = xyz, last = seg class
        else:
            model_points = torch.cat([model_points, torch.cat([part_points, torch.ones((len(part_points), 1), device=part_points.device)*part_idx], dim = 1)], dim = 0)
    
        homog_part_points = torch.cat([part_points, torch.ones((len(part_points), 1)).to(device)], dim = 1)
        part_points = (R.T @ part_points.T).T + T.reshape(-1, 3) #Apply rotation translation to points (think rotation needs to be transposed because of coordinate system differences)
        proj_part_points = (K @ part_points.T).T #Project points to 2D
        coords = proj_part_points[:, :-1]/proj_part_points[:, -1].reshape(-1, 1) #Get 2D points
        coords = torch.clip(coords, 0, args.imsize - 1) #Clip to image boundaries (this is just a check should hopefully not need this)

        #Iterate through each 2D coordinate
        for coord in coords:
            y = int(coord[1])
            x = int(coord[0])
            density_map[y, x] += 1 #Increment count at this 2D coordinate
            seg_density_map[y, x, part_idx] += 1 #Increment count at this 2D coordinate
            canvas[y, x] = 255

    #assert (density_map.sum() == args.n_samples)
    #assert ((seg_density_map.sum(dim=2) == density_map).all())

    #model_points[:, 0] = -model_points[:, 0]
    model_points[:, 1] = -model_points[:, 1]
    #model_points[:, 2] = -model_points[:, 2]

    return density_map, seg_density_map, canvas, model_points

def construct_primitives(part_path):
    verts = None
    faces = None
    num_verts = []
    num_faces = []
    
    num_parts = len(glob(os.path.join(part_path, "new-*.obj"), recursive=True))
    original = False
    if num_parts == 0:
        num_parts = len(glob(os.path.join(part_path, "original-*.obj"), recursive=True))
        original = True

    assert num_parts > 0


    area_probs = torch.zeros((num_parts, 1))
    part_idx_counter = 0
    counter = 0
    while counter < num_parts:
        if original:
            if "original-{}.obj".format(part_idx_counter) in os.listdir(part_path):
                part_verts, part_face_data, aux_data = load_obj(os.path.join(part_path, "original-{}.obj".format(part_idx_counter)))
                part_idx_counter += 1
            else:
                part_idx_counter += 1
                continue
        else:
            if "new-{}.obj".format(part_idx_counter) in os.listdir(part_path):
                part_verts, part_face_data, aux_data = load_obj(os.path.join(part_path, "new-{}.obj".format(part_idx_counter)))
                part_idx_counter += 1
            else:
                part_idx_counter += 1
                continue

        part_area, _ = mesh_face_areas_normals(part_verts, part_face_data.verts_idx)
        area_probs[counter] = part_area.sum()
        if verts is None:
            verts = part_verts
            faces= part_face_data.verts_idx
        else:
            verts = torch.cat([verts, part_verts], dim = 0)
            faces = torch.cat([faces, part_face_data.verts_idx + sum(num_verts)], dim = 0)
        num_verts.append(len(part_verts))
        num_faces.append(len(part_face_data.verts_idx))
        counter += 1

    area_probs = area_probs/area_probs.sum()
    verts = verts - verts.mean(dim=0) #center

    #scale
    scale = torch.sqrt(torch.max(torch.square(verts).sum(dim = 1))) * 1.5
    verts = verts/scale

    #Coordinate System Conversion
    verts[:, 0] = -verts[:, 0]
    verts[:, 1] = -verts[:, 1]

    mesh = Meshes(
                verts=verts.unsqueeze(0).to(torch.float32),
                faces=faces.unsqueeze(0).to(torch.float32),
                textures=TexturesVertex((torch.zeros_like(verts) + torch.tensor([0.7, 0.7, 1.0])).unsqueeze(0))
            ).to(device)

    return mesh, verts, faces, num_verts, num_faces, area_probs


    
def get_annotations(args):
    dataset_path = args.dataset_path
    anno_id = args.anno_id
    output_dir = args.output_dir
    create_if_empty(output_dir)
    curr_im_idx = args.curr_im_idx

    mesh, verts, faces, num_verts, num_faces, area_probs = construct_primitives(os.path.join(dataset_path, f'{anno_id}', 'objs'))

    mesh_out = os.path.join(output_dir, str(anno_id), "objs")
    create_if_empty(mesh_out)

    renderer, R, T = get_mesh_renderer(args)

    #Render images
    images, fragments = renderer(mesh.extend(args.n_views))
    #Get raw depth values from fragments
    depth_values = fragments.zbuf.to(device)

    #Create other output directories
    #image_out = os.path.join(output_dir, str(anno_id), "rgb")
    image_out = os.path.join(output_dir,"rgb")
    create_if_empty(image_out)

    #canny_out = os.path.join(output_dir, str(anno_id), "canny")
    canny_out = os.path.join(output_dir, "canny")
    create_if_empty(canny_out)

    #seg_density_out = os.path.join(output_dir, str(anno_id), "seg_density")
    seg_density_out = os.path.join(output_dir, "seg_density")
    create_if_empty(seg_density_out)

    #points_out = os.path.join(output_dir, str(anno_id), "pcd")
    points_out = os.path.join(output_dir, "pcd")
    create_if_empty(points_out)

    #Save canny images
    for view_idx in range(args.n_views):
        canny = get_canny(args, renderer, verts, faces, num_verts, num_faces, view_idx)
        plt.imsave(os.path.join(canny_out, "{:05d}.png".format(view_idx)), canny)

    #return curr_im_idx + args.n_views
    

    #Save rgb and depth images
    for i in range(len(images)):
        plt.imsave(os.path.join(image_out, "{:08d}.png".format(curr_im_idx + i)), np.uint8(images[i, :, :, :-1].cpu().numpy()*255.0))
        depth = get_normed_depth_from_zbuf(depth_values[i, :, :, 0])
        plt.imsave(os.path.join(depth_out, "{:05d}.png".format(i)), np.uint8(np.dstack([depth, depth, depth])*255.0))

    fx, fy = args.f, args.f
    cx, cy = args.imsize//2, args.imsize//2
    K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy],[0.0, 0.0, 1.0]]).to(device)

    counter = 0
    for i in range(args.n_views):
        density_map, seg_density_map, canvas, labeled_pcd = get_densities(args, verts, faces, num_verts, num_faces, area_probs, rot, trans, K)
        #density_map = (density_map - density_map.min())/(density_map.max() - density_map.min())
        #plt.imsave(os.path.join(points_out, "{:05d}.png".format(counter)), np.uint8(canvas))
        np.save(os.path.join(density_out, "{:05d}.npy".format(counter)), density_map.cpu().numpy())
        np.save(os.path.join(seg_density_out, "{:08d}.npy".format(curr_im_idx + counter)), seg_density_map.cpu().numpy().astype('float16'))
        if counter == 0:
           np.save(os.path.join(points_out, "{:08d}.npy".format(curr_im_idx//args.n_views)), labeled_pcd.cpu().numpy().astype('float16'))
        density_map = (density_map - density_map.min())/(density_map.max() - density_map.min())
        #plt.imsave(os.path.join(points_out, "density_{}.png".format(curr_im_idx + counter)), cv2.applyColorMap(np.uint8(255*density_map), cv2.COLORMAP_JET))
        counter += 1
    return curr_im_idx + args.n_views

def main(args):
    with open("final_lines.txt", 'r') as f:
        lines = f.read().splitlines()
        curr_im_idx = 0
        for i, line in enumerate(tqdm(lines)):
            if i == 0:
                continue
            args.anno_id = line.split(" ")[0]
            args.curr_im_idx = curr_im_idx
            #print("Processing {}...".format(args.anno_id), curr_im_idx//args.n_views, curr_im_idx)
            dirs = ['rgb', 'canny', 'seg_density']
            final_output_path = "/home/niviru/Desktop/FinalProject/partnet_dataset/outputs2"
            for d in dirs:
                curr_dir = os.path.join(args.output_dir, str(args.anno_id),  d)
                out_dir = os.path.join(final_output_path, d)
                counter = ((i - 1)*args.n_views)
                if d == "seg_density":
                    ims = sorted(glob(os.path.join(curr_dir, "*.npy"), recursive=True))
                else:
                    ims = sorted(glob(os.path.join(curr_dir, "*.png"), recursive=True))
                for im in ims:
                    replace_path = os.path.join(out_dir, "{:08d}.".format(counter) + im[-3:])
                    counter += 1
                    shutil.copyfile(im, replace_path)


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--n_views", type=int, default=1)
    parser.add_argument("--n_classes", type=int, default=24)
    parser.add_argument("--f", type=float, default=1000.0)
    parser.add_argument("--imsize", type=int, default=512)
    parser.add_argument("--n_samples", type=int, default=16824)
    parser.add_argument("--dataset_path", type=str, default="/home/niviru/Desktop/FinalProject/partnet_dataset/data/data_v0")
    parser.add_argument("--output_dir", type=str, default="/home/niviru/Desktop/FinalProject/partnet_dataset/outputs")
    parser.add_argument("--anno_id", type=int, default=173)

    args = parser.parse_args()
    main(args)
    
