# Ref: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

import cv2
import pafy
import youtube_dl
import time
import os
from urllib.error import HTTPError
import configparser
import logging
from libs import log

logger = logging.getLogger("Face-Extractor")

config = configparser.ConfigParser()
config.read("./config.ini")


NUM_FRAMES_TO_EXTRACT = config.getint("frame-extract", "NUM_FRAMES_TO_EXTRACT")
WAIT_PER_FRAME = config.getint("frame-extract", "WAIT_PER_FRAME")
CAPTURE_RESOLUTION_HEIGHT= config.getint("frame-extract", "CAPTURE_RESOLUTION_HEIGHT")
CAPTURE_RESOLUTION_WIDTH = config.getint("frame-extract", "CAPTURE_RESOLUTION_WIDTH")
FRAME_SAVE_DIR = config.get("frame-extract", "FRAME_SAVE_DIR")


def check_live(vPafy_object: pafy) -> bool:
    """
    Check whether or not a video URL iss a livestream or not 
    """


def load_video_stram(youtube_stream_url: str):

    try:
        logger.debug(f"Loading Youtube Stream:", youtube_stream_url)
        vPafy = pafy.new(youtube_stream_url)

        logger.debug("Checeking if Stream is live")

        isLive = False
        if vPafy.duration == '00:00:00': isLive = True
            
        print (f"Stream Title: {vPafy.title}")
        logger.debug(f"Is Steam Live: {isLive}")

        if not isLive: raise ValueError

        return vPafy

    except HTTPError as err:
        print("HTTP Error", err.errno)
    except ValueError:
        print("The Youtube video stream is not live!")
    

def main():

    logger.info("Starting Youtube Live Face Detector.")

    vPafy = load_video_stram(
        config.get("frame-extract", "LIVESTREAM_VIDEO_URL")
    )

    # Create frame output directory
    if not os.path.exists(FRAME_SAVE_DIR):
        os.makedirs(FRAME_SAVE_DIR)

    play = vPafy.getbest(preftype="mp4")
    cap = cv2.VideoCapture(play.url) 

    # Assign video capture resosltuion
    cap.set(3, CAPTURE_RESOLUTION_WIDTH)
    cap.set(4, CAPTURE_RESOLUTION_HEIGHT)

    for i in range(NUM_FRAMES_TO_EXTRACT):

        logger.debug(f"Capturing frame {i}/{NUM_FRAMES_TO_EXTRACT}")
        ret, frame = cap.read()
        
        frame_name = (f"frame_{i}.jpg")
        savepath = FRAME_SAVE_DIR + frame_name

        logger.info(f"Saving {frame_name} ---> {i}/{NUM_FRAMES_TO_EXTRACT}")
        cv2.imwrite(savepath, frame)

        logger.debug(f"Waiting for for {WAIT_PER_FRAME}s to load next frame")
        time.sleep(WAIT_PER_FRAME)

    cap.release()

    logger.info(f"Extracted {NUM_FRAMES_TO_EXTRACT} frames successfully")

    logger.info("Done.")

if __name__=="__main__":

    logger.setLevel(logging.INFO)
    log.add_stream_handler(logger)

    try:
        main()

    except Exception as error:
        logger.exception("Unhandled exception:")
        raise error
