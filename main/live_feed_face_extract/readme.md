#### Basic Idea: Input Youtube video stream URL and extract faces
URL and number of frames can be adjusted in `./config.ini`; 
Can also adjust the frame capture resolution. As well as a number of variables
<br>
Paths right now are relative to running in this dir ../live_feed_face_extract

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
    /bin/bash ./extract_detect.sh
    ```
-  To Clean Up: 
    ```
    docker image rm -f youtube-live-frame-extractor
    ```


Docker Image Filestructure:
```
-  config.ini
-  libs
    - log.py
-  face_detection.py
-  frame_extractor.py
-  extract_detect.sh
-  requirements.txt
-  data
```

Todo:
-  Add Testing
-  Improve Filestructure
-  Be able to pass CLI to frame_extractor/face_detector?