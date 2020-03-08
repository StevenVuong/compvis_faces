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
    
    for i in range(NUM_FRAMES_TO_EXTRACT):

        ret, frame = cap.read() # Capture frame-by-frame

        # Assign video capture resosltuion
        ret = cap.set(3, CAPTURE_RESOLUTION_WIDTH)
        ret = cap.set(4, CAPTURE_RESOLUTION_HEIGHT)

        if ret==True:

            print(f"Processing frame {i}/{NUM_FRAMES_TO_EXTRACT}")

            # Save frames
            frame_name = "frame_{0}d.jpg".format(i)
            cv2.imwrite(os.path.join(FRAME_SAVE_DIR, frame_name), frame)

            time.sleep(WAIT_PER_FRAME)

        else:
            break


    cap.release()

    print(f"Extracted {NUM_FRAMES_TO_EXTRACT} frames successfully")


