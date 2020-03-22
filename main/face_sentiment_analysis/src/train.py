# Standard Imports
import logging
import numpy as np
import argparse
import configparser
import cv2
import datetime
import os

# Tensorflow Imports
import tensorflow as tf
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

logger = logging.getLogger("TRAIN")

config = configparser.ConfigParser()
config.read("../config.ini")

BASE_DIR = config["emotion-classify"]["DATA_BASEPATH"]
TRAIN_DIR = os.path.join(BASE_DIR, "images", "test")

BATCH_SIZE = config.getint("emotion-classify", "BATCH_SIZE")
NUM_EPOCH = config.getint("emotion-classify", "NUM_EPOCH")
WIDTH, HEIGHT = np.repeat(
    config.getint("emotion-classify", "IMG_WIDTH_HEIGHT"), 
    2)

def tf_model():

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
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

def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()


class MyCustomCallback(tf.keras.callbacks.Callback):

  def on_train_batch_begin(self, batch, logs=None):
    print('Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

  def on_train_batch_end(self, batch, logs=None):
    print('Training: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))


def main():

    logger.info("Starting " + __name__)

    ## Test for Now; will later run with train
    logger.info("Loading Data")
    y_train = np.load(os.path.join(TRAIN_DIR, 'y_test.npy'))

    datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=5,
        width_shift_range=0.2,
        height_shift_range=0.2
    )

    train_generator = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(WIDTH, HEIGHT),
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        class_mode="categorical",
        shuffle=True
    )

    ### To check on; may have to move article;
    ## Change directorry strucutre!
    # To write..
    model = tf_model()
    print("Train Generator")
    model_info = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(os.listdir(TRAIN_DIR))-1 / 32, 
        epochs=1,
        verbose=1,
        callbacks=[MyCustomCallback()])

    print("done")

    plot_model_history(model_info)


    print("Hello")


if __name__=="__main__":

    logger.setLevel(logging.INFO)
    log.add_stream_handler(logger)

    try:
        main()

    except Exception as error:
        logger.exception("Unhandled exception:")
        raise error
