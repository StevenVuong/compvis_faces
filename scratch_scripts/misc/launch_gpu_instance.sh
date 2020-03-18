#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

gcloud config set project linear-ether-270414
gcloud config set compute/zone europe-west4-b

gcloud compute instances create gpu-instance-deeplab \
    --machine-type n1-standard-4 --zone europe-west4-c \
    --accelerator type=nvidia-tesla-v100,count=1 \
    --scopes=storage-full --boot-disk-size=200GB \
    --image-family tf-latest-gpu --image-project deeplearning-platform-release \
    --maintenance-policy TERMINATE --restart-on-failure \

# Once Initiated, SSH into instance and run install for Nvidia drivers