"""
## Extract face and crop out
## Replace original Image with Augment tag
## Augment recursively those with the augment tag

Ref: https://www.tensorflow.org/tutorials/images/data_augmentation
"""
from PIL import Image

import sys
sys.path.append("..")
from libs.face_box import detect_face, FaceBbox

#facebbox_obj = FaceBbox(detect_face(img))
def main():

    print("Augmenting Images")


if __name__=="__main__":
    main()