# Compvis Playground
Incorporate different aspects of computer vision and deep learning, with (improving) engineering practices to learn and showcase as a computer vision repository

#### Current Idea:
-  Take input feed of video data and extract frames
-  Detect faces within the frames (face detection) and crop faces out
-  Increase Resolution of the cropped faces:
    -  https://github.com/xinntao/ESRGAN
    -  GAN? https://www.tensorflow.org/tutorials/generative/pix2pix
    -  https://github.com/idealo/image-super-resolution 
-  Determine whose face it is (face recognition) 
    -  Train by augmenting input data of a single person (training)
    -  Single image of each news presenter (setup); augment then run
-  Face sentiment analysis
-  Do analysis matching of face and sentiment? bwahaha!

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
-  Set Project with `gcloud config set project my-project`
