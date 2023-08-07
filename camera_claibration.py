import cv2
import numpy as np
# Use stereo calibration to get both matrices:
# retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2 = cv2.stereoCalibrate() # need to learn how to use this function

# Extract from image EXIF data (approximate):
img = cv2.imread('data/nerf_llff_data/RedCar/images/IMG_2077.jpeg')
mtx, dist = cv2.estimateCameraParametersFromEXIF(img)

# Assume no distortion
h, w = img.shape[:2]
mtx = np.array([[w, 0, w/2], 
                [0, h, h/2],
                [0, 0, 1]])
dist = np.zeros((4,1))