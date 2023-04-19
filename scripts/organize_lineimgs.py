import os
import json
import shutil
import argparse
from glob import glob
from collections import defaultdict

from tqdm import tqdm

def parse_args():
    
    parser = argparse.ArgumentParser('Organize_LineImgs', add_help=False)
    
    parser.add_argument('--img_dir', type=str, default='../dataset/shapenet_rotate_edge')
    parser.add_argument('--orig_dir', type=str, default='../dataset/r2n2_shapenet_dataset/r2n2/ShapeNetRendering/03001627')
    parser.add_argument('--orig_line_dir', type=str, default=None)
    # parser.add_argument('--orig_line_dir', type=str, default='../dataset/r2n2_shapenet_original_line')
    parser.add_argument('--out_dir', type=str, default='../dataset/r2n2_shapenet_dataset/r2n2/LineDrawings/03001627')
    parser.add_argument('--split_path', type=str, default='../dataset/r2n2_shapenet_dataset/line_split.json')
    
    return parser.parse_args()

def main(args):
    files = glob(f"{args.img_dir}/*", recursive=True)
    files.sort()

    for file in tqdm(files, total=len(files)):
        ## parse name of image
        obj_name = os.path.basename(file)
        
        ## create output directory
        outdir = os.path.join(args.out_dir, obj_name, "rendering")
        os.makedirs(outdir, exist_ok=True)
        dic = {}
        
        for count, img in enumerate(glob(f"{file}/*.png")):
            name = os.path.basename(img)
            dic[count+10] = name[:-4]
            out_name = "%02d.png" % (count+10)
            
            ## move image
            shutil.copy(img, os.path.join(outdir, out_name))
            
            if count == 4:
                name = os.path.basename(img)
                dic[count+10] = name[:-4]
                out_name = "%02d.png" % (count+10)
                
                ## move image
                shutil.copy(img, os.path.join(outdir, out_name))
            
                ## save mapping as json file
                with open(os.path.join(outdir, "name_map.json"), "w") as outdic:
                    json.dump(dic, outdic, indent=4)
                
                ## move line images of original data
                if args.orig_line_dir is not None:
                    orig_imgs = glob(f"{args.orig_line_dir}/{obj_name}/*.png")
                    for orig in orig_imgs:
                        if 'x' in orig:
                            continue
                        shutil.copy(orig, f"{outdir}/{os.path.basename(orig)}")

                ## move additional data
                src = os.path.join(args.orig_dir, obj_name, "rendering")
                shutil.copy(os.path.join(src, "feats.npy"), os.path.join(outdir, "feats.npy"))
                shutil.copy(os.path.join(src, "rendering_metadata.txt"), os.path.join(outdir, "rendering_metadata.txt"))
                shutil.copy(os.path.join(src, "renderings.txt"), os.path.join(outdir, "renderings.txt"))
        
    ## create split json file
    traindic = defaultdict(list)
    testdic = defaultdict(list)

    dirs = glob(f"{args.out_dir}/*")
    train_num = int(len(dirs) * 0.8)
    for i, dir in enumerate(dirs):
        obj_name = os.path.basename(os.path.normpath(dir))
        
        imgs = glob(f"{dir}/**/*.png", recursive=True)
        for img in imgs:
            f = os.path.basename(img)
            
            if i < train_num:
                traindic[obj_name].append(f[:-4])
            else:
                testdic[obj_name].append(f[:-4])

    finaldic = {
        "train": {
            "03001627": traindic
        },
        "test": {
            "03001627": testdic
        }
    }

    with open(args.split_path, "w") as outfile:
        json.dump(finaldic, outfile, indent=4)


    # objs = glob("/home/ubuntu/IKEA/dataset/r2n2_shapenet_dataset/r2n2/LineDrawings/03001627/*")

    # ls = []
    # for obj in objs:
    #     if not os.path.exists(os.path.join(obj, "rendering", "feats.npy")):
    #         print(obj)
    #         ls.append(obj)
    # print("@@@ incomplete objs: ", len(ls))
    
if __name__ == "__main__":
    args = parse_args()
    main(args)