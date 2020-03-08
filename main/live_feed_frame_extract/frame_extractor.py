# Ref: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

import cv2
import pafy
import youtube_dl
import time
import os
from urllib.error import HTTPError
import configparser


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
    isLive = False
    if vPafy.duration == '00:00:00': isLive = True

    print (f"Stream Title: {vPafy.title}")

    return isLive


if __name__=="__main__":

    # Load Video Stream
    try:
        vPafy = pafy.new(
            config.get("frame-extract", "LIVESTREAM_VIDEO_URL")
            )
    
        isLive = check_live(vPafy)
        print(f"isLive: {isLive}")

        if not isLive: raise ValueError

    except HTTPError as err:
        print("HTTP Error", err.errno)
    except ValueError:
        print("The Youtube video stream is not live!")


    # Create frame output directory
    if not os.path.exists(FRAME_SAVE_DIR):
        os.makedirs(FRAME_SAVE_DIR)


    play = vPafy.getbest(preftype="mp4")
    cap = cv2.VideoCapture(play.url) 

    # Assign video capture resosltuion
    cap.set(3, CAPTURE_RESOLUTION_WIDTH)
    cap.set(4, CAPTURE_RESOLUTION_HEIGHT)

    for i in range(NUM_FRAMES_TO_EXTRACT):

        ret, frame = cap.read() # Capture frame-by-frame

        frame_name = (f"frame_{i}.jpg")
        savepath = FRAME_SAVE_DIR + frame_name
        print(f"Saving {frame_name} ---> {i}/{NUM_FRAMES_TO_EXTRACT}")

        cv2.imwrite(savepath, frame)

        time.sleep(WAIT_PER_FRAME)

    cap.release()

    print(f"Extracted {NUM_FRAMES_TO_EXTRACT} frames successfully")


