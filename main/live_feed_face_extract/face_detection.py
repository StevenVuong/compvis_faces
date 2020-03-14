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


class Face:
    """
    Class to parse face_json output from MTCNN output.
    """
    def __init__(self, face_json: dict):

        self.confidence = face_json["confidence"]
        self.box_coords = self.get_box_coords(face_json["box"])

    def get_box_coords(self, box_array: list) -> list:

        x1, y1, width, height = box_array
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        return [x1, x2, y1, y2]


def main():

    logger.info("Starting Frame Face Detection.")

    for frame in os.listdir(FRAMES_DIR)[2:3]: # ONLY DO ONE IMG FOR NOW
        frame_path = os.path.join(FRAMES_DIR, frame)
    
        logger.debug(f"Loading {frame_path}")
        img = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)

        faces_json = detect_face(img)
        logger.info(f"Detected {len(faces_json)} in {frame}")

        # now loop thorugh faces; extract one by one
        # maybe save in subfoldeer per image? Need a naming convention..
        # image_faceno?
        # Tofix: Color; reload im?
        for face_json in faces_json:
            
            logger.debug("Creating Face class")
            face = Face(face_json)

            if face.confidence < FACE_CONFIDENCE_THERSHOLD:
                continue
            
            x1, x2, y1, y2 = (face.box_coords)
            cropped_face = img[y1:y2, x1:x2]

            extracted_frames_path = os.path.join(FACE_OUTPUT_DIR, frame)
            if not os.path.exists(FACE_OUTPUT_DIR):
                os.makedirs(FACE_OUTPUT_DIR)
            cv2.imwrite(extracted_frames_path, cropped_face)


    logger.info("Completed Face Detection")

if __name__=="__main__":

    logger.setLevel(logging.INFO)
    log.add_stream_handler(logger)

    try:
        main()

    except Exception as error:
        logger.exception("Unhandled exception:")
        raise error
