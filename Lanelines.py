import cv2
import matplotlib.image as mpimg
import numpy as np


class LaneLines:
    def __init__(self):
        self.left_fit = None
        self.right_fit = None
        self.binary = None
        self.nonzero = None
        self.nonzerox = None
        self.nonzeroy = None
        self.clear_visibility = True

        self.dir = []

        self.left_curve_img = mpimg.imread('left_turn.png')
        self.right_curve_img = mpimg.imread('right_turn.png')
        self.keep_straight_img = mpimg.imread('straight.png')

        self.left_curve_img = cv2.normalize(
            src=self.left_curve_img,
            dst=None, alpha=0, beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U
        )

        self.right_curve_img = cv2.normalize(
            src=self.right_curve_img,
            dst=None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U
        )

        self.keep_straight_img = cv2.normalize(
            src=self.keep_straight_img,
            dst=None, alpha=0, beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U
        )

        self.nwindows = 9
        self.margin = 100
        self.minpix = 50

    def extract_features(self, my_photo):
        self.img = my_photo
        self.window_height = np.int(my_photo.shape[0]//self.nwindows)
        self.nonzero = my_photo.nonzero()
        self.nonzerox = np.array(self.nonzero[1])
        self.nonzeroy = np.array(self.nonzero[0])
