"""
## Extract face and crop out
## Replace original Image with Augment tag
## Augment recursively those with the augment tag

Better practice is to make a tfds dataset to augment on the fly.
For our purpose we just augment and save to make more images


Important: Run from <pwd>/facenet/ dir
    -   python src/new_contrib/augment_data.py 

Todo:
    -  Improve pathing

Ref: 
    -  https://note.nkmk.me/en/python-pillow-rotate/
    -  https://note.nkmk.me/en/python-pillow-flip-mirror/
"""
from PIL import Image, ImageOps
import os
import tensorflow as tf
import numpy as np

import sys
sys.path.append("../")
from libs.face_box import detect_face, FaceBbox

IMAGE_DIR = "../../data/facenet/images"
TRAIN_IMAGE_DIR = os.path.join(IMAGE_DIR, "train_raw")
TEST_IMAGE_DIR = os.path.join(IMAGE_DIR, "test_raw")
#facebbox_obj = FaceBbox(detect_face(img))


def main():

    print("Augmenting Images")

    # Build path to Image
    image_foldernames = os.listdir(TRAIN_IMAGE_DIR)
    for image_foldername in image_foldernames:

        image_folder_path = os.path.join(TRAIN_IMAGE_DIR, image_foldername)
        if not os.path.isdir(image_folder_path):
            continue

        for image in os.listdir(image_folder_path):
            if not image.endswith((".jpg", ".png", ".jpeg")):
                continue
            image_path = os.path.join(image_folder_path, image)
            
            # Need to extract face here and make save to path
            # Question: Do we delete the original image?

            # Open image and perform augmentations
            image = Image.open(image_path)

            rotated_image_a = image.rotate(7)
            rotated_image_b = image.rotate(7)

            mirror_image = ImageOps(image)

            # Make new dir with extarcteed face and applied rotations


            break


if __name__=="__main__":
    main()