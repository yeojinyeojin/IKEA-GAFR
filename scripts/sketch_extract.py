import os
import json
import shutil


def delete_and_create(path):
    if os.path.exists(path):
        raise ValueError(f'Delete folder {path}')
    os.mkdir(path)
    return True

def find_img_name(ind_hashmap, class_name, model_name, page_id):
    # refactor later
    found = False
    for k, v in ind_hashmap.items():
        if v['class_name'] == class_name \
                and v['model_name'] == model_name \
                and v['pdf_page_id'] + 1 == page_id:
            found = True
            return k
    if found is False:
        ## due to 1.pdf issue
        for k, v in ind_hashmap.items():
            if v['class_name'] == class_name \
                    and v['model_name'] == model_name \
                    and v['model_page_id'] + 1 == page_id + 2:
                print(f'using modified images for {class_name, model_name, page_id}')
                return k

    raise ValueError(f'missing class and model info for {class_name, model_name, page_id}')


if __name__ == '__main__':
    with open("../dataset/ikea_man/main_data.json", 'r') as f:
        main_json = json.load(f)

    with open("../dataset/ind_map.json", 'r') as f:
        ind_hashmap = json.load(f)

    delete_and_create("../dataset/images_masks")
    plain_sketch_hashmap = {}

    for model_dict in main_json:
        class_name = model_dict['category']
        model_name = model_dict['name']
        for step_dict in model_dict['steps']:
            page_id, local_step_id = step_dict['page_id'], step_dict['step_id']
            mask_path = f'../dataset/ikea_man/mask/{class_name}/{model_name}/step_{local_step_id}_mask.png'

            try:
                img_name = find_img_name(ind_hashmap, class_name, model_name, page_id)

                new_img_name = f'{img_name[:-4]}_step_{local_step_id}.png'
                plain_sketch_hashmap[new_img_name] = {
                    'class_name': class_name,
                    'model_name': model_name,
                    'page_id': page_id - 1,  # 0-indexed
                    'local_step_id': local_step_id,
                }
                dst_path = f'../dataset/images_masks/{new_img_name}'
                shutil.copyfile(src=mask_path, dst=dst_path)
            except Exception as e:
                print(f'missing class and model info for {class_name, model_name, page_id}')

    plain_sketch_hashmap = dict(sorted(plain_sketch_hashmap.items()))
    with open("../dataset/images_masks.json", "w") as f:
        json.dump(plain_sketch_hashmap, f, indent=6)