#!/bin/bash

set -euo pipefail

# docker build and run commands
docker build . -t undistort_test
docker run undistort_test python undistort_cv2.py