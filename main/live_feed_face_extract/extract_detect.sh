#!/bin/bash

set -euo pipefail

python ./frame_extractor.py

python ./face_detection.py