"""
Extract every 10th frame (using ffmpeg) to an output directory
Then sample the images according to a Gaussian Distribution, deleting everything else
"""

import os, errno, subprocess, sys
import numpy as np

# Configurables
input_video_dir = "/Users/steven/Downloads/semseg_vid_extract/input"
output_dir = "/Users/steven/Downloads/semseg_vid_extract/output"
frame_rate = 10
num_images_output = 10 + 1 # Set to 1 more than desired

# read sys args
if sys.argv[1] != None:
    num_images_output = int(sys.argv[1]) + 1

if sys.argv[2] != None:
    output_dir = str(sys.argv[2])

def run_ffmpeg(input_path, output_path, frame_rate=10):
    subprocess.run(
    [r"ffmpeg", "-i", input_path, "-q:v","2","-vf",
    f"fps={frame_rate}", output_path + "/%06d.jpeg",
    "-hide_banner"], check=True
    )


def sample_frames_ndist(video_filepath, mean=0, stdev=0.2):

    extracted_frames = os.listdir(video_filepath)
    num_extracted_frames = len(extracted_frames)
    all_unique = False

    while all_unique==False:
        # Normal dist, mean=0, stdev=0.1, 20 images
        norm_dist = np.random.normal(mean, stdev, num_images_output)

        # Scale normal dist and round
        norm_dist += abs(min(norm_dist))
        range_normal_dists = max(norm_dist) - min(norm_dist)
        norm_dist *= num_extracted_frames / range_normal_dists
        norm_dist = np.around(norm_dist)

        uniques = np.unique(norm_dist, return_counts=True)
        # Check there are no duplicates
        if len(uniques[1]) == num_images_output:
            all_unique=True

    return np.array(norm_dist)


def create_dir(video_filepath):
    if not os.path.exists(video_output_filepath):
        try:
            os.makedirs(video_output_filepath)
        except OSError as e:
            if e.errno != errno.EEXIST:
                # Directory already exists
                pass


if __name__=="__main__":

    for video_name in os.listdir(input_video_dir):

        video_input_filepath = os.path.join(input_video_dir, video_name)
        video_output_filepath = os.path.join(output_dir, video_name)
        
        # Create output directory
        create_dir(video_output_filepath)

        # Run Ffmpeg
        run_ffmpeg(video_input_filepath, video_output_filepath, frame_rate)

        # Sample non-identical normally distributed frames
        norm_dist_indexes = sample_frames_ndist(video_output_filepath)
        
        # Delete unwanted frames
        for idx,frame in enumerate(sorted(os.listdir(video_output_filepath))):
            if idx not in norm_dist_indexes:
                os.remove(os.path.join(video_output_filepath, frame))

        print(f"Extracted {num_images_output-1} normally distributed frames from {video_name}")