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
TRAIN_DIR = os.path.join(BASE_DIR, "images", "train")
TEST_DIR = os.path.join(BASE_DIR, "images", "test")

# BATCH_SIZE = config.getint("emotion-classify", "BATCH_SIZE")
# NUM_EPOCH = config.getint("emotion-classify", "NUM_EPOCH")
BATCH_SIZE = 10
NUM_EPOCH = 5
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
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
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

  def on_test_batch_begin(self, batch, logs=None):
    print('Evaluating: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

  def on_test_batch_end(self, batch, logs=None):
    print('Evaluating: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))


class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
  """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

  def __init__(self, patience=0):
    super(EarlyStoppingAtMinLoss, self).__init__()

    self.patience = patience

    # best_weights to store the weights at which the minimum loss occurs.
    self.best_weights = None

  def on_train_begin(self, logs=None):
    # The number of epoch it has waited when loss is no longer minimum.
    self.wait = 0
    # The epoch the training stops at.
    self.stopped_epoch = 0
    # Initialize the best as infinity.
    self.best = np.Inf

  def on_epoch_end(self, epoch, logs=None):
    current = logs.get('loss')
    if np.less(current, self.best):
      self.best = current
      self.wait = 0
      # Record the best weights if current results is better (less).
      self.best_weights = self.model.get_weights()
    else:
      self.wait += 1
      if self.wait >= self.patience:
        self.stopped_epoch = epoch
        self.model.stop_training = True
        print('Restoring model weights from the end of the best epoch.')
        self.model.set_weights(self.best_weights)

  def on_train_end(self, logs=None):
    if self.stopped_epoch > 0:
      print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))


def scheduler(epoch):
    """
    This function keeps the learning rate at 0.001 for the first ten epochs
    and decreases to 0.0001 after
    """
    if epoch < 10:
        return 0.001
    else:
        return 0.0001 


class LearningRateScheduler(tf.keras.callbacks.Callback):
  """Learning rate scheduler which sets the learning rate according to schedule.

  Arguments:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current learning rate
          as inputs and returns a new learning rate as output (float).
  """

  def __init__(self):
    super(LearningRateScheduler, self).__init__()
    self.schedule = None

  def on_epoch_begin(self, epoch, logs=None):
    if not hasattr(self.model.optimizer, 'lr'):
      raise ValueError('Optimizer must have a "lr" attribute.')
    # Get the current learning rate from model's optimizer.
    lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
    # Call schedule function to get the scheduled learning rate.
    self.schedule = scheduler(epoch)
    scheduled_lr = self.schedule(epoch, lr)
    # Set the value back to the optimizer before this epoch starts
    tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
    print('\nEpoch %05d: Learning rate is %6.4f.' % (epoch, scheduled_lr))



def main():

    logger.info("Starting " + __name__)

    logger.info("Creating Train Datagen")
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=5,
        width_shift_range=0.2,
        height_shift_range=0.2,
        validation_split=0.05 # Validation Split
    )

    logger.info("Creating Train Generator")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(WIDTH, HEIGHT),
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        class_mode="categorical",
        shuffle=True
    )

    logger.info("Creating Validation Generator")
    validation_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, # same directory as training data
        target_size=(WIDTH, HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation") # set as validation data

    logger.info("Loading TF Model")
    model = tf_model()

    logger.info("Training Model")
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data = validation_generator,
        validation_steps = validation_generator.samples // BATCH_SIZE,
        epochs=NUM_EPOCH,
        verbose=1,
        callbacks=[
            MyCustomCallback(), 
            EarlyStoppingAtMinLoss(),
            LearningRateScheduler()])

    print("Done")

    print(history.history.keys())
    #plot_model_history(history)


if __name__=="__main__":

    logger.setLevel(logging.INFO)
    log.add_stream_handler(logger)

    try:
        main()

    except Exception as error:
        logger.exception("Unhandled exception:")
        raise error
