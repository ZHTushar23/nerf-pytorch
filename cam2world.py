import cv2
import numpy as np

def calibrate_camera(calibration_images, pattern_size):
    obj_points = []  # 3D points in real world space
    img_points = []  # 2D points in image plane

    # Prepare 3D points for the calibration pattern (e.g., checkerboard)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    for image_file in calibration_images:
        img = cv2.imread(image_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find corners of the calibration pattern
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            obj_points.append(objp)
            img_points.append(corners)

    # Perform camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    return mtx, dist

def estimate_camera_pose(mtx, dist, scene_image, pattern_size):
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    img = cv2.imread(scene_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        ret, rvecs, tvecs = cv2.solvePnP(objp, corners, mtx, dist)

        # Get the camera-to-world transformation matrix
        R, _ = cv2.Rodrigues(rvecs)
        T = np.hstack((R, tvecs))

        return T

    else:
        print("Calibration pattern not found in the scene image.")
        return None

if __name__ == "__main__":
    # Calibration parameters
    calibration_images = ["data/nerf_llff_data/RedCar/images/IMG_2077.jpeg", "data/nerf_llff_data/RedCar/images/IMG_2078.jpeg", 
                          "data/nerf_llff_data/RedCar/images/IMG_2079.jpeg"]
    pattern_size = (9, 6)  # Number of inner corners in the calibration pattern

    # Scene image for camera pose estimation
    scene_image = "scene_image.jpg"

    # Camera calibration
    mtx, dist = calibrate_camera(calibration_images, pattern_size)

    # Camera pose estimation
    camera_to_world_matrix = estimate_camera_pose(mtx, dist, scene_image, pattern_size)

    if camera_to_world_matrix is not None:
        print("Camera-to-world transformation matrix:")
        print(camera_to_world_matrix)
