import json
import os
import argparse as ap

CATEGORIES = {"Bench": 0, "Chair": 1, "Desk": 2, "Misc": 3, "Shelf": 4, "Table": 5}

def create_if_empty(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return True


if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="../dataset")
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=700)
    args = parser.parse_args()

    labels_dir_path = os.path.join(args.dataset_path, "labels")

    with open("../annotator/ikea_od_two.json", 'r') as f:
        vgg_annots = json.load(f)

    with open(os.path.join(args.dataset_path, "ind_map.json"), 'r') as f:
    #with open("../dataset/ind_map.json", 'r') as f:
        ind_hashmap = json.load(f)

    start_id = args.start_id  # inclusive
    end_id = args.end_id  # inclusive

    img_metadata = vgg_annots['_via_img_metadata']
    create_if_empty(labels_dir_path)

    for _img_name_cat, img_info in img_metadata.items():
        img_name_prefix = img_info['filename'][:-4]
        if start_id <= int(img_name_prefix) <= end_id:
            #txt_path = f'{labels_dir_path}/{img_name_prefix}.txt'
            txt_path = os.path.join(labels_dir_path, img_name_prefix + ".txt")

            img_ind_hashmap = ind_hashmap[f'{img_name_prefix}.png']
            img_w, img_h = img_ind_hashmap['img_w'], img_ind_hashmap['img_h']
            label_sequences = []
            category_idx = CATEGORIES[img_ind_hashmap["class_name"]]
            for region_dict in img_info['regions']:
                class_label = region_dict['region_attributes']['sketch_type']
                shape_dict = region_dict['shape_attributes']

                class_num = 0 if class_label == 'skip' else 1
                width = shape_dict['width']
                height = shape_dict['height']
                x_center = shape_dict['x'] + width // 2
                y_center = shape_dict['y'] + height // 2

                _x = x_center/img_w
                _y = y_center/img_h
                _w = width/img_w
                _h = height/img_h
                obj_text = f'{category_idx} {class_num} {_x:06f} {_y:06f} {_w:06f} {_h:06f}'
                label_sequences.append(obj_text)

            with open(txt_path, 'w') as f:
                for _label_seq in label_sequences:
                    f.writelines(_label_seq)
                    f.writelines('\n')
