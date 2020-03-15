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
    -  Big Refactor, very hacky right now

Ref: 
    -  https://note.nkmk.me/en/python-pillow-rotate/
    -  https://note.nkmk.me/en/python-pillow-flip-mirror/
"""
import cv2
import os
import tensorflow as tf
import numpy as np
from scipy import ndimage, misc

import sys
sys.path.append("../")
from libs.face_box import detect_face, FaceBbox, save_img

IMAGE_DIR = "../../data/facenet/images"
TRAIN_IMAGE_DIR = os.path.join(IMAGE_DIR, "train_raw")
TEST_IMAGE_DIR = os.path.join(IMAGE_DIR, "test_raw")


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
            
            # Open image
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            
            # Get faces json and loop through to extract face
            faces_json = detect_face(image)
            for face_idx, face_json in enumerate(faces_json):

                if face_json["confidence"] < 0.99:
                    continue

                # Extract face
                facebbox_obj = FaceBbox(face_json)
                extracted_face_img = facebbox_obj.extract_face(image)

                # Build path to save
                face_save_path = "{0}_{2}{1}".format(
                    *os.path.splitext(image_path) + (face_idx,)
                )
                # Save to augmented outputs
                face_save_path = face_save_path.replace("train_raw", "train_augmented")
                
                # Save face
                save_img(extracted_face_img, face_save_path)

                # Perform augmentations on faces and save
                rotated_image_l = ndimage.rotate(extracted_face_img, 7, reshape=False)
                rotated_l_save_path = face_save_path.replace(".jpg", "_rotate_l.jpg")
                save_img(rotated_image_l, rotated_l_save_path)

                rotated_image_r = ndimage.rotate(extracted_face_img, -7, reshape=False)
                rotated_r_save_path = face_save_path.replace(".jpg", "_rotate_r.jpg")
                save_img(rotated_image_r, rotated_r_save_path)
                 
                flipped_image = cv2.flip(extracted_face_img, 1)
                flipped_save_path = face_save_path.replace(".jpg", "_flipped.jpg")
                save_img(flipped_image, flipped_save_path)


if __name__=="__main__":
    main()