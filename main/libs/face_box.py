"""
Detects faces in images/frames, extracts them from the image and saves output

Todo: 
    -  Add tests

Refs: 
https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/
https://github.com/ipazc/mtcnn 
"""
from mtcnn import MTCNN
import cv2
import logging
from libs import log
import os
import json
import numpy as np

logger = logging.getLogger("Face-Detector")

def detect_face(img: np.ndarray) -> list:
    """
    Detects faces using Multi-Task CNN (MTCNN)
    Ref: https://github.com/ipazc/mtcnn
    Args:
        img(np.ndarray)
    Returns:
        list of JSON objects 
    """

    logger.debug(f"Initialising MTCNN and detecting faces in frame")
    detector = MTCNN()

    return detector.detect_faces(img)


def save_to_json(json_list: list, save_path: str):
    """
    Saves json list to file with indent 4 and sorted keys
    Args:
        json_list(list)
        save_path(str)
    """

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    logger.debug(f"Writing json to {save_path}")
    with open(save_path, "w") as out_json:
                json.dump(json_list, out_json, indent=4, sort_keys=True)


def save_img(img: np.ndarray, save_path: str):
    """
    Saves image list file 
    Args:
        img(np.ndarray)
        save_path(str)
    """
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    cv2.imwrite(save_path, img)


class FaceBbox:
    """
    Class with functions to parse face_json output from MTCNN output
    and extract faces from images

    Args:
        face_json(dict): dictionary with information about each face

    Attributes:
        confidence(float): confidence [0,1] that the box contains a face
        box_coords(list): pixel coordinatess of box containing face
    """

    def __init__(self, face_json: dict):

        self.confidence = face_json["confidence"]
        self.box_coords = self.get_box_coords(face_json["box"])

    def get_box_coords(self, box_array: list) -> list:
        """
        Ges pixel coordinates for box surrounding persons face

        Args: 
            box_array(list): Value from face_json with x1, y1, width, height of box

        Returns:
            (list): box coordinates of 4 corners of box
        """

        x1, y1, width, height = box_array
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        return [x1, x2, y1, y2]

    def extract_face(self, img:np.ndarray) -> np.ndarray:
        """
        Crop face of an image using box coordinates.
        Also applies operations to convert face to RGB from BGR (cv2 save default)
        As well as resizing the image

        Args: 
            img(np.ndarray): 2d image array

        Returns:
            cropped_face(np.ndarray): Extracted face from the image

        """
        x1, x2, y1, y2 = self.box_coords
        cropped_face = img[y1:y2, x1:x2]

        # cv2 works in bgr by default, convert to rgb
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)

        # resize to deisred size
        cropped_face = cv2.resize(
            cropped_face, (OUTPUT_SIZE,OUTPUT_SIZE), interpolation=cv2.INTER_CUBIC
            )

        return cropped_face
