"""
Goal: 
Build docker image that can run Python with CV2
Run Docker container with mounted volume:
    Input: Dir of distorted frames 
    Output: Dir of undistorted frames 
"""

import numpy as np
import cv2
import configparser
import os

config = configparser.ConfigParser()
config.read("./camera_config.ini")


class CameraSettings:

    def __init__(self):
        FC = np.repeat(
            config["camera_config"]["focal_length"], 2
            ) # Focal Lengths
        CC = [
            int(config["camera_config"]["width"])/2, 
            int(config["camera_config"]["height"])/2
            ] # principle points (usually half width/height)
        KC = [
            float(config["camera_config"]["k1"]), 
            float(config["camera_config"]["k2"]), 
            float(config["camera_config"]["t1"]), 
            float(config["camera_config"]["t2"]), 
            float(config["camera_config"]["k3"])
            ] # distortion coeffs

        self.cam_matrix, self.distortion_profile = self.create_matrix_profile(
            FC, CC, KC
            )
        

    def create_matrix_profile(self, fc, cc, kc):
        """Create the camera matrix and distortion profile.

        Take in th e focal lengths, principle points, and distortion coefficients
        and return the camera matrix and distortion coefficients in the form
        OpenCV needs.
        Takes:
            fc - the x and y focal lengths [focallength_x, focallength_y]
            cc - the x and y principle points [point_x, point_y]
            kc - the distortion coefficients [k1, k2, p1, p2, k3]
        Gives:
            cam_matrix - the camera matrix for the video
            distortion_profile - the distortion profile for the video
        """
        fx, fy = fc
        cx, cy = cc
        cam_matrix = np.array([[fx,  0, cx],
                            [ 0, fy, cy],
                            [ 0,  0,  1]], dtype='float32')
        distortion_profile = np.array(kc, dtype='float32')
        return cam_matrix, distortion_profile


if __name__=="__main__":

    # Get Camera Settings
    CS = CameraSettings()

    # Make output directory
    output_dir = config["dirs"]["output_dir"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    input_dir = config["dirs"]["input_dir"]

    # Undistort each input frame
    for frame in os.listdir(input_dir):

        frame_path = os.path.join(input_dir, frame)

        distorted_img = cv2.imread(frame_path)

        if type(distorted_img) == np.ndarray:

            undistorted_img = cv2.undistort(distorted_img, CS.cam_matrix, CS.distortion_profile)
            # Save output file
            print(os.path.join(output_dir, frame))
            # not saving for some reason?
            cv2.imwrite(os.path.join(output_dir, frame), undistorted_img)
    