# Compvis Playground
Incorporate different aspects of computer vision and deep learning, with (improving) engineering practices to learn and showcase as a computer vision repository

#### Current Idea:
-  Take input feed of video data and extract frames `$(PWD)/main/live_feed_face_extract`
    -  Detect faces within the frames (face detection) and crop faces out 
-  Increase Resolution of the cropped faces (idealo: image-super-resolution) `$(PWD)/jupyter_notebooks/Image_Super_Resolution.ipynb`
-  Determine whose face it is (face recognition) `$(PWD)/main/facenet`
    -  Train by augmenting input data of a single person (training)
    -  Single image of each news presenter (setup); augment then run
-  Face sentiment analysis -> Building a regular CNN `$(PWD)/main/face_sentiment_analysis`
    -  Also visualises heatmap of CNN

SideGoal:

-  **Todo**: Fast Style Transfer -> https://github.com/lengstrom/fast-style-transfer 

#### G-Cloud Auth Starter:
-  Must change gcloud auth logins/outs every time by `gcloud auth login <ENTER_EMAIL_ADDRERSS>`
-  Check account by `gcloud auth list` and change with `gcloud config set account <ACCOUNT>`
-  Confirm by running `gcloud auth list` again and `gcloud projects list`
-  Set Project with `gcloud config set project my-project`
