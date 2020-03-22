# Standard Imports
import logging
import numpy as np
import argparse
import configparser
import cv2
import datetime
import os
import matplotlib.pyplot as plt

# Tensorflow Imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Libs from other dirs
import sys
sys.path.append("../..")
from libs import log

logger = logging.getLogger("TRAIN")

config = configparser.ConfigParser()
config.read("../config.ini")

BASE_DIR = config["emotion-classify"]["DATA_BASEPATH"]
TRAIN_DIR = os.path.join(BASE_DIR, "images", "train")
TEST_DIR = os.path.join(BASE_DIR, "images", "test")

BATCH_SIZE = config.getint("emotion-classify", "BATCH_SIZE")
NUM_EPOCH = config.getint("emotion-classify", "NUM_EPOCH")
WIDTH, HEIGHT = np.repeat(
    config.getint("emotion-classify", "IMG_WIDTH_HEIGHT"), 
    2)
    

def tf_model():

    model = Sequential()

    model.add(Conv2D(16, kernel_size=(2, 2), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(16, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu'))
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(2, 2), activation='relu'))
    model.add(Conv2D(64, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=(2, 2), activation='relu'))
    model.add(Conv2D(256, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=0.0001, decay=1e-6),
        metrics=['accuracy'])

    return model


def main():

    logger.info("Starting " + __name__)

    logger.info("Creating Train Datagen")
    train_datagen = ImageDataGenerator(
        rescale=1./255)

    logger.info("Creating Train Generator")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(WIDTH, HEIGHT),
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        class_mode="categorical",
        shuffle=True
    )

    val_datagen = ImageDataGenerator(
        rescale=1./255)

    logger.info("Creating Val Generator")
    val_generator = val_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(WIDTH, HEIGHT),
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        class_mode="categorical",
        shuffle=True
    )

    logger.info("Loading TF Model")
    model = tf_model()

    logger.info("Training Model")
    history = model.fit_generator(
        train_generator,
        steps_per_epoch = train_generator.samples // BATCH_SIZE,
        epochs = NUM_EPOCH,
        verbose = 1,
        validation_data=val_generator,
        validation_steps=train_generator.samples // BATCH_SIZE,
        callbacks = None)

    print("Done")



if __name__=="__main__":

    logger.setLevel(logging.INFO)
    log.add_stream_handler(logger)

    try:
        main()

    except Exception as error:
        logger.exception("Unhandled exception:")
        raise error
