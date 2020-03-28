"""
Visualise heatmaps
https://towardsdatascience.com/visualizing-intermediate-activation-in-convolutional-neural-networks-with-keras-260b36d60d0s
"""
# Standard Imports
import logging
import numpy as np
import configparser
import os
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import models, preprocessing

# Libs from other dirs
import sys
sys.path.append("../..")
from libs import log

logger = logging.getLogger("Visualise-Activation")

config = configparser.ConfigParser()
config.read("../config.ini")

BASE_DIR = config["emotion-classify"]["DATA_BASEPATH"]
SAMPLE_IM_PATH = os.path.join(BASE_DIR, "images", "predict", "im_A.png")

WIDTH, HEIGHT = np.repeat(
    config.getint("emotion-classify", "IMG_WIDTH_HEIGHT"), 
    2)

def main():

  logger.info("Starting " + __name__)

  logger.info("Loading pre-trained TF CNN Model")
  model =  tf.keras.models.load_model("./model.h5")
  
  logger.info("Creating activation model from CNN top layers")
  # Extracts the outputs of the top 12 layers
  layer_outputs = [layer.output for layer in model.layers[:12]]

  # Creates a model that will return these outputs, given the model input
  activation_model = tf.keras.models.Model(
    inputs=model.input, outputs=layer_outputs)

  logger.info("Loading sample image")
  img = preprocessing.image.load_img(SAMPLE_IM_PATH, target_size=(WIDTH, HEIGHT), grayscale=True)
  img = preprocessing.image.img_to_array(img)/255
  img = np.expand_dims(img, axis=0)

  logger.info(f"Predicting Image {os.path.basename(SAMPLE_IM_PATH)}")
  activations = activation_model.predict(img, batch_size=1)

  # Look at the shape for each layer to get an idea of what is going on 
  layer_activation = activations[1]
  print(layer_activation.shape)

  # View a filter of choice at a particular activation layer 
  # Can view different channels of activation
  plt.matshow(layer_activation[0, :, :, 10], cmap='viridis')
  plt.savefig("./activation.png")

if __name__=="__main__":
  main()