import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# a class that calibrate camera using chessboard images
# attributes:
#   mtx (np.array): Camera matrix 
#   dist (np.array): Distortion coefficients
class CameraCalibration():

    # constructor
    # params:
    #   image_dir (str): path to folder contains chessboard images
    #   nx (int): width of chessboard (number of squares)
    # ny (int): height of chessboard (number of squares)
    def __init__(self, image_dir, nx, ny, debug=False):
        file_names = glob.glob("{}/*".format(image_dir))
        object_points = []
        img_points = []
        
        # Coordinates of chessboard's corners in 3D
        obj_p = np.zeros((nx*ny, 3), np.float32)
        obj_p[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        
        # Go through all chessboard images
        for f in file_names:
            img = mpimg.imread(f)

            # Convert to grayscale image
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Find chessboard corners
            return_val, corners = cv2.findChessboardCorners(img, (nx, ny))
            if return_val:
                img_points.append(corners)
                object_points.append(obj_p)

        shape = (img.shape[1], img.shape[0])
        return_val, self.mtx, self.dist, _, _ = cv2.calibrateCamera(object_points, img_points, shape, None, None)

        if not return_val:
            raise Exception("Unable to calibrate camera")

    # return undistort image
    # inputs:
    #   img (np.array): input image
    # outputs:
    #   img (np.array): undistorted image
    def undistort(self, img):
        # Convert to grayscale image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
