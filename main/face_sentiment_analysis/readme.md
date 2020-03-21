# Face Emotion Analysis!

This is a supervised learning task to classify faces into 6 emotions:
-  Anger
-  Disgust
-  Fear
-  Happy
-  Sad
-  Surprise
-  Neutral

So we will try to build a UNet (not so great? UNet usually for semseg.. loll)
with Skip connections to achieve this; and show
activation heat map layers in order to achieve this task.

### Quickstart
-  `python3 -m pip install -r requirements.txt`
-  `python main.py`


### References:
-  Dataset Provider: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
-  Helper: https://github.com/gitshanks/fer2013 To work with Numpy Array
-  Image Data Generator Method: https://raw.githubusercontent.com/atulapra/Emotion-detection/master/src/emotions.py
