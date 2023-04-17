import os
import json
import shutil
from glob import glob
from collections import defaultdict

from tqdm import tqdm

ORIG_DIR = '../dataset/r2n2_shapenet_dataset/r2n2/ShapeNetRendering/03001627'
ORIG_LINE_DIR = '../dataset/r2n2_shapenet_original_line'
IMG_DIR = '../dataset/shapenet_rotate_edge'
OUT_DIR = '../dataset/r2n2_shapenet_dataset/r2n2/LineDrawings/03001627'
JSON_FILE = '../dataset/r2n2_shapenet_dataset/line_split.json'

files = glob(f"{IMG_DIR}/*", recursive=True)
files.sort()
# files = files[f9:] #first object only has 8 images

for file in tqdm(files, total=len(files)):
    ## parse name of image
    obj_name = os.path.basename(file)
    
    ## create output directory
    outdir = os.path.join(OUT_DIR, obj_name, "rendering")
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
            orig_imgs = glob(f"{ORIG_LINE_DIR}/{obj_name}/*.png")
            for orig in orig_imgs:
                if 'x' in orig:
                    continue
                shutil.copy(orig, f"{outdir}/{os.path.basename(orig)}")

            ## move additional data
            src = os.path.join(ORIG_DIR, obj_name, "rendering")
            shutil.copy(os.path.join(src, "feats.npy"), os.path.join(outdir, "feats.npy"))
            shutil.copy(os.path.join(src, "rendering_metadata.txt"), os.path.join(outdir, "rendering_metadata.txt"))
            shutil.copy(os.path.join(src, "renderings.txt"), os.path.join(outdir, "renderings.txt"))
        
    # if count == 10:
    #     dic[count+9] = out_name[:-5]
    #     out_name = "%02d.png" % (count+9)
        
    #     ## move image
    #     shutil.copy(file, os.path.join(outdir, out_name))
        
    #     ## save mapping as json file
    #     with open(os.path.join(outdir, "name_map.json"), "w") as outdic:
    #         json.dump(dic, outdic, indent=4)
        
    #     ## reset
    #     dic, count = {}, 1
        
    #     ## move line images of original data
    #     imgs = glob(f"{ORIG_LINE_DIR}/{obj_name}/*.png")
    #     for img in imgs:
    #         if 'x' in img:
    #             continue
    #         shutil.copy(img, f"{outdir}/{os.path.basename(img)}")
        
    #     ## move additional data
    #     src = os.path.join(ORIG_DIR, obj_name, "rendering")
    #     shutil.copy(os.path.join(src, "feats.npy"), os.path.join(outdir, "feats.npy"))
    #     shutil.copy(os.path.join(src, "rendering_metadata.txt"), os.path.join(outdir, "rendering_metadata.txt"))
    #     shutil.copy(os.path.join(src, "renderings.txt"), os.path.join(outdir, "renderings.txt"))
    
    # else:
    #     dic[count+9] = out_name[:-5]
    #     out_name = "%02d.png" % (count+9)
    #     count += 1
        
    #     ## move image
    #     shutil.copy(file, os.path.join(outdir, out_name))
    
## create split json file
traindic = defaultdict(list)
testdic = defaultdict(list)

dirs = glob(f"{OUT_DIR}/*")
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

with open(JSON_FILE, "w") as outfile:
    json.dump(finaldic, outfile, indent=4)


objs = glob("/home/ubuntu/IKEA/dataset/r2n2_shapenet_dataset/r2n2/LineDrawings/03001627/*")

ls = []
for obj in objs:
    if not os.path.exists(os.path.join(obj, "rendering", "feats.npy")):
        print(obj)
        ls.append(obj)
print("@@@ incomplete objs: ", len(ls))