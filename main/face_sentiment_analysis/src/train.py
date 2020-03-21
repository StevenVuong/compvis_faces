# Standard Imports
import logging
import numpy as np
import argparse
import configparser
import cv2
import os

# Tensorflow Imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Libs from other dirs
import sys
sys.path.append("../..")
from libs import log

logger = logging.getLogger(__name__)

config = configparser.ConfigParser()
config.read("../config.ini")

BASE_DIR = config["emotion-classify"]["DATA_BASEPATH"]
TRAIN_CSVPATH = os.path.join(BASE_DIR, "train.csv")

BATCH_SIZE = config.getint("emotion-classify", "BATCH_SIZE")
NUM_EPOCH = config.getint("emotion-classify", "NUM_EPOCH")


def main():

    logger.info("Starting " + __name__)

    train_datagen = ImageDataGenerator(
        rescale=1/.255,
        horizontal_flip=True
    )

    # Something to look into
    # https://keras.io/preprocessing/image/
    train_generator = train_datagen.flow_from_directory(
        TRAIN_CSVPATH,
        target_size=(48,48),
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        class_mode="categorical"
    )

    print("Hello")


if __name__=="__main__":

    logger.setLevel(logging.INFO)
    log.add_stream_handler(logger)

    try:
        main()

    except Exception as error:
        logger.exception("Unhandled exception:")
        raise error
