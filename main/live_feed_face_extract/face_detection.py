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

def main():

    logger.info("Starting Frame Face Detection.")

    for frame in os.listdir(FRAMES_DIR):
        frame_path = os.path.join(FRAMES_DIR, frame)

        logger.debug(f"Loading {frame}")
        img = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)

        logger.debug(f"Initialising MTCNN and detecting faces")
        detector = MTCNN()

        faces_json = detector.detect_faces(img)

        if not os.path.exists(FACES_JSON_DIR):
            os.makedirs(FACES_JSON_DIR)

        logger.debug(f"Outputting json files to... <DIR>")
        with open(FACES_JSON_DIR + os.path.basename(frame) + ".json", "w") as out_json:
            json.dump(faces_json, out_json)


if __name__=="__main__":

    logger.setLevel(logging.INFO)
    log.add_stream_handler(logger)

    try:
        main()

    except Exception as error:
        logger.exception("Unhandled exception:")
        raise error
