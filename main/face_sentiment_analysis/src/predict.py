# Standard Imports
import logging
import numpy as np
import configparser
import os

# Tensorflow Imports
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Libs from other dirs
import sys
sys.path.append("../..")
from libs import log

logger = logging.getLogger("PREDICT")

config = configparser.ConfigParser()
config.read("../config.ini")

BASE_DIR = config["emotion-classify"]["DATA_BASEPATH"]
PRED_DIR = os.path.join(BASE_DIR, "images", "predict")

WIDTH, HEIGHT = np.repeat(
    config.getint("emotion-classify", "IMG_WIDTH_HEIGHT"), 
    2)

emotion_dict = {
    0: "Angry", 
    1: "Disgusted", 
    2: "Fearful", 
    3: "Happy", 
    4: "Neutral", 
    5: "Sad", 6: 
    "Surprised"}

def main():

    logger.info("Starting " + __name__)

    logger.info("Loading pre-trained TF CNN Model")
    model =  tf.keras.models.load_model("./model.h5")

    logger.info("Evaluating Results")
    for img_name in os.listdir(PRED_DIR):

        if not img_name.endswith((".jpg", ".png", ".jpeg")):
            continue

        logger.info(f"Loading Image {img_name}")

        img_path = os.path.join(PRED_DIR, img_name)

        img = image.load_img(img_path, target_size=(WIDTH, HEIGHT), grayscale=True)
        img = image.img_to_array(img)/255
        img = np.expand_dims(img, axis=0)

        logger.info(f"Predicting Image {img_name}")
        result = model.predict(img, batch_size=1)
        result = np.argmax(result)

        emotion = emotion_dict[result]
        logger.info(f"{img_name} has emotion {emotion}")

    logger.info("Done")


if __name__=="__main__":

    logger.setLevel(logging.INFO)
    log.add_stream_handler(logger)

    try:
        main()

    except Exception as error:
        logger.exception("Unhandled exception:")
        raise error
