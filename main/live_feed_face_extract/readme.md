#### Basic Idea: Input Youtube video stream URL and extract faces
URL and number of frames can be adjusted in `./config.ini`; 
default is Sky News Youtube livestream and 5 frames with 20 second interval

#### Docker Quickstarter

-  cd to this directory
-  To build docker image: 
    ```
    docker build -t youtube-live-face-extractor .
    ```
-  Running Docker Image with Mounted Data Volume
    ```
    docker run -v $(pwd)/../../data:/usr/src/app/data \
    youtube-live-face-extractor \
    python ./frame_extractor.py
    ```
-  To Clean Up: 
    ```
    docker image rm -f youtube-live-frame-extractor
    ```


Docker Image Filestructure:
```
-  config.ini
-  frame_extractor.py
-  requirements.txt
-  data
```