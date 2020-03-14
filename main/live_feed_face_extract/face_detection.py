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
import configparser
import logging
from libs import log
import os
import json
import numpy as np

logger = logging.getLogger("Face-Detector")

config = configparser.ConfigParser()
config.read("./config.ini")

FRAMES_DIR = config.get("frame-extract", "FRAME_SAVE_DIR")
FACES_JSON_DIR = config.get("face-detect", "JSON_OUTPUT_DIR")
FACE_CONFIDENCE_THERSHOLD = config.getfloat(
    "face-detect", "FACE_CONFIDENCE_THRESHOLD"
    )
FACE_OUTPUT_DIR = config.get("face-detect", "FACE_OUTPUT_DIR")
OUTPUT_SIZE = config.getint("face-detect", "OUTPUT_SIZE")


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


class Face:
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


def main():

    logger.info("Starting Frame Face Detection.")

    for frame in os.listdir(FRAMES_DIR):

        frame_name = frame.split(".")[0]
        frame_path = os.path.join(FRAMES_DIR, frame)
    
        logger.debug(f"Loading {frame_path}")
        # cv2 automatically loads as BGR, must invert
        img = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)

        faces_json = detect_face(img)
        logger.info(f"Detected {len(faces_json)} faces in {frame}")


        for idx,face_json in enumerate(faces_json):
            
            logger.debug("Creating Face class")
            face = Face(face_json)

            if face.confidence < FACE_CONFIDENCE_THERSHOLD:
                logger.info(f"Face {frame_name}_{idx} confidence too low!")
                continue
            
            logger.debug(f"Extracting Face {idx} from {frame_name}")
            cropped_face = face.extract_face(img)

            extracted_frames_path = os.path.join(
                FACE_OUTPUT_DIR, frame_name+f"_{idx}.jpg"
            )
            logger.debug("Saving Extracted Frame")
            save_img(cropped_face, extracted_frames_path)

    logger.info("Completed Face Detection")


if __name__=="__main__":

    logger.setLevel(logging.INFO)
    log.add_stream_handler(logger)

    try:
        main()

    except Exception as error:
        logger.exception("Unhandled exception:")
        raise error
