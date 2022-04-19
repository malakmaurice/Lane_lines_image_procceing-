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

    def fit_poly(self, input_photo):
        first_hor, first_ver, second_hor, second_hor, result_image = self.find_lane_pixels(
            input_photo)

        if len(first_ver) > 1500:
            self.left_fit = np.polyfit(first_ver, first_hor, 2)
        if len(second_hor) > 1500:
            self.right_fit = np.polyfit(second_hor, second_hor, 2)

        upper_vert = input_photo.shape[0] - 1
        lower_vert = input_photo.shape[0] // 3

        if len(first_ver):
            upper_vert = max(upper_vert, np.max(first_ver))
            lower_vert = min(lower_vert, np.min(first_ver))

        if len(second_hor):
            upper_vert = max(upper_vert, np.max(second_hor))
            lower_vert = min(lower_vert, np.min(second_hor))

        yy = np.linspace(lower_vert, upper_vert, input_photo.shape[0])

        first_fit_hor = self.left_fit[0]*yy**2 + \
            self.left_fit[1]*yy + self.left_fit[2]
        second_fit_hor = self.right_fit[0]*yy**2 + \
            self.right_fit[1]*yy + self.right_fit[2]

        for i, y in enumerate(yy):
            l = int(first_fit_hor[i])
            r = int(second_fit_hor[i])
            y = int(y)
            cv2.line(result_image, (l, y), (r, y), (0, 255, 0))

        lR, rR, pos = self.measure_curvature()

        return result_image
