# Standard Imports
import logging
import numpy as np
import argparse
import configparser
import cv2
import os

# Tensorflow Imports
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.models import model_from_json

# Libs from other dirs
import sys
sys.path.append("../..")
from libs import log

logger = logging.getLogger("TRAIN")

config = configparser.ConfigParser()
config.read("../config.ini")

BASE_DIR = config["emotion-classify"]["DATA_BASEPATH"]
TRAIN_CSVPATH = os.path.join(BASE_DIR, "train.csv")

BATCH_SIZE = config.getint("emotion-classify", "BATCH_SIZE")
NUM_EPOCH = config.getint("emotion-classify", "NUM_EPOCH")


def main():

    logger.info("Starting " + __name__)

    logger.info("Loading Data")
    y_train = np.load(os.path.join(BASE_DIR, 'y.npy'))
    y_test = np.load(os.path.join(BASE_DIR, 'y.npy'))

    ## Make Data Generators... Etc..
    # Train on test images for now


    print("Hello")


if __name__=="__main__":

    logger.setLevel(logging.INFO)
    log.add_stream_handler(logger)

    try:
        main()

    except Exception as error:
        logger.exception("Unhandled exception:")
        raise error
