# Compvis Playground
Incorporate different aspects of computer vision and deep learning, with (improving) engineering practices to learn and showcase as a computer vision repository

#### Current Idea:
-  Take input feed of video data
-  Recognise faces from that video data (face recognition)

    -  https://github.com/davidsandberg/facenet (First Approach)
    -  Second approach is to use object detection to get a person, extract them then take out their face
        -  https://medium.com/@somaniswastik/face-recognition-using-tensorflow-pre-trained-model-opencv-91184efa4aaf
        -  https://colab.research.google.com/github/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb#scrollTo=DWh_1zz6aqxs
-  Then zoom into face pixel
-  Increase Resolution:
    -  https://github.com/xinntao/ESRGAN
    -  GAN? https://www.tensorflow.org/tutorials/generative/pix2pix
    -  https://github.com/idealo/image-super-resolution 
-  Face sentiment analysis

SideGoal:

-  Visualise Heat Map of a CNN
    -  https://github.com/idealo/cnn-exposed
    -  `/code_samples/sophie_vis_activation_snippet.py`
-  Fast Style Transfer: https://github.com/lengstrom/fast-style-transfer
-  Have Separate folder structure for each idea; perhaps they can link in. And feed into/out of a data folder with download instructions (to put into a bash script?)

#### G-Cloud Auth Starter:
-  Must change gcloud auth logins/outs every time by `gcloud auth login <ENTER_EMAIL_ADDRERSS>`
-  Check account by `gcloud auth list` and change with `gcloud config set account <ACCOUNT>`
-  Confirm by running `gcloud auth list` again and `gcloud projects list`