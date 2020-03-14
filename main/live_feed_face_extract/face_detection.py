from mtcnn import MTCNN
import cv2
import configparser
import logging
from libs import log
import os
import json

logger = logging.getLogger("Face-Detector")

config = configparser.ConfigParser()
config.read("./config.ini")

FRAMES_DIR = config.get("frame-extract", "FRAME_SAVE_DIR")
FACES_JSON_DIR = config.get("face-detect", "JSON_OUTPUT_DIR")


def detect_face(frame_path: str) -> list:
    """
    Detects faces using Multi-Task CNN (MTCNN)
    Ref: https://github.com/ipazc/mtcnn
    Args:
        frame_path(str)
    Returns:
        list of JSON objects 
    """
    
    logger.debug(f"Loading {frame_path}")
    img = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)

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


def main():

    logger.info("Starting Frame Face Detection.")

    for frame in os.listdir(FRAMES_DIR)[0:2]: # ONLY DO ONE IMG FOR NOW
        frame_path = os.path.join(FRAMES_DIR, frame)

        faces_json = detect_face(frame_path)

        save_path = FACES_JSON_DIR + frame.split(".")[0] + ".json"
        save_to_json(faces_json, save_path)

        ## Need to Process to get above threshold certainty (config) ##
        # Then get box dims and extract the face
        # Then save face

    logger.info("Completed Face Detection")

if __name__=="__main__":

    logger.setLevel(logging.INFO)
    log.add_stream_handler(logger)

    try:
        main()

    except Exception as error:
        logger.exception("Unhandled exception:")
        raise error
