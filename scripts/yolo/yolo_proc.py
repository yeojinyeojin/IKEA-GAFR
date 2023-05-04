import pathlib
import os
import cv2
import numpy as np


IMAGES_DIR = '../../dataset/images'
LABELS_DIR = '../../dataset/labels'


if __name__ =='__main__':
      file_names = [126, 382]  # '382', '00222'
      img_list = []
      for file_name in file_names:
            file_name = f'{file_name:05d}'
            img = cv2.imread(os.path.join(IMAGES_DIR, f'{file_name}.png'))
            with open(os.path.join(LABELS_DIR, f'{file_name}.txt'), 'r') as f:
                  boxes = f.readlines()

            img_h, img_w = img.shape[:2]
            for box in boxes:
                  _, class_num, *bbox = box.split()
                  bbox = list(map(float, bbox))
                  x, y, w, h = bbox
                  x, w = int(x * img_w), int(w * img_w)
                  y, h = int(y * img_h), int(h * img_h)

                  x -= w//2
                  y -= h//2

                  # img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 5)
                  # cv2.putText(img, f'class_{class_num}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                  x1, y1, x2, y2 = x, y, x + w, y + h
                  color = (0, 0, 255) if class_num == '1' else (255, 0, 0)  # (200, 200, 255)
                  text_color = (255, 255, 255)
                  font_thickness = 9
                  font_size = 5
                  label = f'class_{class_num}'
                  # For bounding box
                  img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 5)

                  # For the text background
                  # Finds space required by the text so that we can put a background with that amount of width.
                  (w, h), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thickness)

                  # Prints the text.
                  img = cv2.rectangle(img, (x1, y1 + h), (x1 + w, y1), color, -1)
                  img = cv2.putText(img, label, (x1, y1 - 5 + h),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, font_thickness)

            img_list.append(img)
            cv2.imwrite(f'{file_name}.png', img)

      merged_images = np.concatenate(img_list, axis=1)
      cv2.imwrite(f'train_batch1_new.png', merged_images)
