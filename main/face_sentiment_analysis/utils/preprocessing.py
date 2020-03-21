import pandas as pd
import numpy as np
import os
import configparser
import logging
from progressbar import ProgressBar, Percentage, Bar

config = configparser.ConfigParser()
config.read("../config.ini")

# Libs from other dirs
import sys
sys.path.append("../..")
from libs import log

logger = logging.getLogger("Preprocessing")


BASE_DIR = config["emotion-classify"]["DATA_BASEPATH"]
DATA_CSVPATH = os.path.join(BASE_DIR, "fer2013/fer2013.csv")
WIDTH, HEIGHT = np.repeat(
    config.getint("emotion-classify", "IMG_WIDTH_HEIGHT"), 
    2)


def main():

    logger.info("Starting Preprocessing")

    data = pd.read_csv(DATA_CSVPATH)
    pixels_lists = data['pixels'].tolist()

    # Get features for training
    X = []
    pbar = ProgressBar(widgets=[Percentage(), Bar(">")])
    for pixels_list in pbar(pixels_lists):
        # Parse string into xy array
        xx = [int(pixels_list) for pixels_list in pixels_list.split(' ')]
        xx = np.asarray(xx).reshape(WIDTH, HEIGHT)
        X.append(xx.astype('float32'))

    X = np.asarray(X)
    X = np.expand_dims(X, -1)
    logger.info(f"Number of examples in dataset: {str(len(X))}")
    logger.info(f"Number of Features: {str(len(X[0]))}")

    # Get labels for training (one-hot encoded)
    y = pd.get_dummies(data['emotion']).values
    logger.info(f"Number of Labels: {str(len(y[0]))}")

    logger.info("Preprocessing done, saving as .npy")
    np.save(os.path.join(BASE_DIR, 'X'), X)
    np.save(os.path.join(BASE_DIR, 'y'), y)
    logger.info(f"X, y stored in {BASE_DIR}")

    logger.info("Done.")


if __name__=="__main__":

    logger.setLevel(logging.INFO)
    log.add_stream_handler(logger)

    try:
        main()

    except Exception as error:
        logger.exception("Unhandled exception:")
        raise error

