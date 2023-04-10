import os
import PIL
import json
import pathlib

from pdf2image import convert_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)


def create_if_empty(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return True


if __name__ == '__main__':
    pdf_paths = sorted(list(pathlib.Path('../dataset/ikea_man/pdfs').glob('**/*.pdf')))
    create_if_empty("../dataset/images")

    ind_hashmap = {}
    running_ctr = 0

    _prev_class_name = ''
    _prev_model_name = ''
    _prev_page_end = 0
    for pdf_path in pdf_paths:
        *_, class_name, model_name = str(pdf_path.parent).rsplit('/')
        images = convert_from_path(pdf_path)

        if class_name != _prev_class_name or model_name != _prev_model_name:
            _prev_page_end = 0

        for i, img in enumerate(images):
            w, h = img.size
            img_name = f"{running_ctr:05d}.png"
            img.save(f"../dataset/images/{img_name}")
            ind_hashmap[img_name] = {
                "class_name": class_name,
                "model_name": model_name,
                "pdf_path": str(pdf_path)[3:],
                "img_h": h,
                "img_w": w,
                "pdf_page_id": i,
                "model_page_id": i + _prev_page_end,
                "pdf_name": pdf_path.name,
            }
            running_ctr += 1

        _prev_class_name = class_name
        _prev_model_name = model_name
        _prev_page_end = i + 1

    with open("../dataset/ind_map.json", "w") as f:
        json.dump(ind_hashmap, f, indent=6)