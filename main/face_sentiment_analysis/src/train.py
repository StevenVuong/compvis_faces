# Standard Imports
import logging
import numpy as np
import configparser
import os

# Tensorflow Imports
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import cnn_model

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
    
model = cnn_model()


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

    logger.info("Loading TF CNN Model")
    model = tf_model()

    logger.info("Training Model")
    history = model.fit_generator(
        train_generator,
        steps_per_epoch = train_generator.samples // BATCH_SIZE,
        epochs = NUM_EPOCH,
        verbose = 1,
        validation_data = val_generator,
        validation_steps = val_generator.samples // BATCH_SIZE,
        callbacks = None)

    logger.info("Training Completed!")
    results = model.evaluate_generator(
        generator = val_generator,
        steps = val_generator.samples)
    logger.info('test loss, test acc:', results)

    #  Need to export model and histsory (Make Callback?)
    #  Then have history plot in another function; Save Plot Image somewhere
    #  And a predict class as well as evaluate function
    #  Then do some heatmap vissualisation to see how our CNN performs
    #  Todo: Add callbacks use config.yaml instead of ini
    #  Be able to call tensorboard during training?

    print("Done")



if __name__=="__main__":

    logger.setLevel(logging.INFO)
    log.add_stream_handler(logger)

    try:
        main()

    except Exception as error:
        logger.exception("Unhandled exception:")
        raise error
