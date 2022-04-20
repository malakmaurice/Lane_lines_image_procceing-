import cv2
import numpy as np


def threshold_rel(my_photo, start, end):
    lower = np.min(my_photo)
    upper = np.max(my_photo)

    t1 = lower + (upper - lower) * start
    t2 = lower + (upper - lower) * end
    return np.uint8((my_photo >= t1) & (my_photo <= t2)) * 255


def threshold_abs(my_photo, lower, upper):
    return np.uint8((my_photo >= lower) & (my_photo <= upper)) * 255


class Thresholding:
    def __init__(self):
        pass

    # a method that extracts pixels related to each other in an image
    def forward(self, img):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h_channel = hls[:, :, 0]
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]
        v_channel = hsv[:, :, 2]

        right_lane = threshold_rel(l_channel, 0.8, 1.0)
        right_lane[:, :750] = 0

        left_lane = threshold_abs(h_channel, 20, 30)
        left_lane &= threshold_rel(v_channel, 0.7, 1.0)
        left_lane[:, 550:] = 0

        img2 = left_lane | right_lane

        return img2
