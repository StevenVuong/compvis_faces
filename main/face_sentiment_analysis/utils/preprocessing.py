"""
Load the Image dataset; split into train and test sets; save as images
"""
import pandas as pd
import numpy as np
import os
import configparser
import logging
from PIL import Image
from sklearn.model_selection import train_test_split
from progressbar import ProgressBar

config = configparser.ConfigParser()
config.read("../config.ini")

# Libs from other dirs
import sys
sys.path.append("../..")
from libs import log

logger = logging.getLogger("Preprocessing")


BASE_DIR = config["emotion-classify"]["DATA_BASEPATH"]
TRAIN_TEST_SPLIT = config.getfloat("emotion-classify", "TRAIN_TEST_SPLIT")
WIDTH, HEIGHT = np.repeat(
    config.getint("emotion-classify", "IMG_WIDTH_HEIGHT"), 
    2)
IMAGE_SAVE_DIR = os.path.join(BASE_DIR, "images")
DATA_CSVPATH = os.path.join(BASE_DIR, "fer2013/fer2013.csv")


def main():

    logger.info("Starting Preprocessing")

    # Make directories to save, according to the label
    if not os.path.exists(IMAGE_SAVE_DIR):
        for i in range(7):
            os.makedirs(os.path.join(IMAGE_SAVE_DIR, "test", str(i)))
            os.makedirs(os.path.join(IMAGE_SAVE_DIR, "train", str(i)))

    logger.info("Loading Data")

    data = pd.read_csv(DATA_CSVPATH)

    pixels_lists = data["pixels"].values
    y = data['emotion'].values

    logger.info(f"Number of Samples in dataset: {len(pixels_lists)}")
    logger.info(f"Number of Labels: {len(y)}")

    logger.info("Splitting data into train and test sets")
    pixels_lists_train, pixels_lists_test, y_train, y_test = train_test_split(
        pixels_lists, y, train_size=TRAIN_TEST_SPLIT)

    logger.info(f"Training Size: {len(y_train)}, Test Size: {len(y_test)}")

    logger.info("Converting and Saving Training Images")
    pbar = ProgressBar()
    for idx, pixels_list in pbar(enumerate(pixels_lists_train)):

        # Parse string into xy array
        xx = [int(pixels_list) for pixels_list in pixels_list.split(' ')]
        xx = np.asarray(xx).reshape(WIDTH, HEIGHT)

        emotion_label = y_train[idx]

        # Save as Image
        im = Image.fromarray(xx.astype("uint8"))
        im.save(f"{IMAGE_SAVE_DIR}/train/{emotion_label}/thumb_{idx:04d}.png")

    pbar = ProgressBar()
    logger.info("Converting and Saving Test Images")
    for idx, pixels_list in pbar(enumerate(pixels_lists_test)):

        # Parse string into xy array
        xx = [int(pixels_list) for pixels_list in pixels_list.split(' ')]
        xx = np.asarray(xx).reshape(WIDTH, HEIGHT)

        emotion_label = y_test[idx]

        # Save as Image
        im = Image.fromarray(xx.astype("uint8"))
        im.save(f"{IMAGE_SAVE_DIR}/test/{emotion_label}/thumb_{idx:04d}.png")


    logger.info("Done.")


if __name__=="__main__":

    logger.setLevel(logging.INFO)
    log.add_stream_handler(logger)

    try:
        main()

    except Exception as error:
        logger.exception("Unhandled exception:")
        raise error

