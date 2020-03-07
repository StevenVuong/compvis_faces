#!/bin/bash

set -euo pipefail

# gsutil cp ...

## Get for each video; output every 10th frame into a folder of it's name
# for vid_file in ./batch_one_vids/*.mp4; do
#     vid_foldername=${vid_file%.*}
#     mkdir -p $vid_foldername
#     # Image every 10th frame
#     ffmpeg -i $vid_file -vf fps=10 $vid_foldername/out%05d.png
#     #echo $d
# done

# Loop through directories 
for frames_dir in ./batch_one_vids/*/; do
    echo $frames_dir
    python3 ./useful_scripts/get_n_evenlyspaced_images.py $frames_dir
done

# rename files
# for frames_dir in ./batch_one_vids/*/; do
#     python3 ./useful_scripts/rename_file.py $frames_dir
# done