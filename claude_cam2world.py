import cv2
import numpy as np

img1_rgb = cv2.imread('data/nerf_llff_data/RedCar/images/IMG_2077.jpeg')
img2_rgb = cv2.imread('data/nerf_llff_data/RedCar/images/IMG_2078.jpeg')

# Convert to grayscale
img1 = cv2.cvtColor(img1_rgb, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2_rgb, cv2.COLOR_BGR2GRAY)


# Detect SIFT features
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None)  
kp2, des2 = sift.detectAndCompute(img2,None)

# Match features 
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Filter matches
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)

# Get points        
pts1 = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
pts2 = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)


# Compute camera matrices 
F, masks = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_8POINT)
# M1, M2 = cv2.computeCorrespondEpilines(pts1, pts2, F)

# compute projection matrices for each camera.
U, S, Vt = np.linalg.svd(F)
W = np.array([[0,-1,0], [1,0,0], [0,0,1]])

M2 = np.concatenate((U, U[:,2].reshape(3,1)), axis=1) #M2 is the 3x4 projection matrix for camera 2
M1 = np.concatenate((Vt.T, Vt.T[:,2].reshape(3,1)), axis=1) #M1 is the 3x4 projection matrix for camera 1



# Invert and create camera to world matrix
cameraMatrix1,rotMat1, transVect1,_,_,_,_ = cv2.decomposeProjectionMatrix(M1)

# print(rotMat1.shape, transVect1.shape)

cameraMatrix2,rotMat2, transVect2,_,_,_,_ = cv2.decomposeProjectionMatrix(M2)
# print(rotMat2.shape, transVect2.shape)
# print(transVect1)
# print(transVect2)

# print("Camera Matrix:--------------------")
# print(cameraMatrix1)
# print(cameraMatrix2)

# Convert 4x1 homogenous translation vector to 3x1 vector
T1 = transVect1[:3]/transVect1[3]
T2 = transVect2[:3]/transVect2[3]

# T11 = cv2.convertPointsFromHomogeneous(transVect1.reshape(-1,1,4)).reshape(3,-1)
# print(T11,T1)

#We invert the rotation and translation like:
R1_inv = rotMat1.T
t1_inv = -R1_inv @ T1

R2_inv = rotMat2.T 
t2_inv = -R2_inv @ T2

# Then reconstruct the full camera to world transformations:
T1_world = np.hstack((R1_inv, t1_inv.reshape(3,1)))
T2_world = np.hstack((R2_inv, t2_inv.reshape(3,1)))

print(T1_world)
print(T2_world)

