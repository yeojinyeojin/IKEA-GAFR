import json
import os

def create_if_empty(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return True


if __name__ == '__main__':
    labels_dir_path = "../dataset/labels"

    with open("../annotator/ikea_od_two.json", 'r') as f:
        vgg_annots = json.load(f)

    with open("../dataset/ind_map.json", 'r') as f:
        ind_hashmap = json.load(f)

    start_id = 0  # inclusive
    end_id = 700  # inclusive

    img_metadata = vgg_annots['_via_img_metadata']
    create_if_empty(labels_dir_path)

    for _img_name_cat, img_info in img_metadata.items():
        img_name_prefix = img_info['filename'][:-4]
        if start_id <= int(img_name_prefix) <= end_id:
            txt_path = f'{labels_dir_path}/{img_name_prefix}.txt'

            img_ind_hashmap = ind_hashmap[f'{img_name_prefix}.png']
            img_w, img_h = img_ind_hashmap['img_w'], img_ind_hashmap['img_h']
            label_sequences = []
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
                obj_text = f'{class_num} {_x:06f} {_y:06f} {_w:06f} {_h:06f}'
                label_sequences.append(obj_text)

            with open(txt_path, 'w') as f:
                for _label_seq in label_sequences:
                    f.writelines(_label_seq)
                    f.writelines('\n')