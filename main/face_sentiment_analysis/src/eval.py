# Standard Imports
import logging
import numpy as np
import configparser
import os

# Tensorflow Imports
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Libs from other dirs
import sys
sys.path.append("../..")
from libs import log

logger = logging.getLogger("EVAL")

config = configparser.ConfigParser()
config.read("../config.ini")

BASE_DIR = config["emotion-classify"]["DATA_BASEPATH"]
TEST_DIR = os.path.join(BASE_DIR, "images", "test")

BATCH_SIZE = config.getint("emotion-classify", "BATCH_SIZE")
NUM_EPOCH = config.getint("emotion-classify", "NUM_EPOCH")
WIDTH, HEIGHT = np.repeat(
    config.getint("emotion-classify", "IMG_WIDTH_HEIGHT"), 
    2)


def main():

    logger.info("Starting " + __name__)

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

    logger.info("Loading pre-trained TF CNN Model")
    model =  tf.keras.models.load_model("./model.h5")

    logger.info("Evaluating Results")
    results = model.evaluate_generator(
        generator = val_generator,
        steps = val_generator.samples)

    print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
    print(f"eval loss: {results[0]}")
    print(f"eval accuracy: {results[1]}")
    print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")

    logger.info("Done")


if __name__=="__main__":

    logger.setLevel(logging.INFO)
    log.add_stream_handler(logger)

    try:
        main()

    except Exception as error:
        logger.exception("Unhandled exception:")
        raise error
