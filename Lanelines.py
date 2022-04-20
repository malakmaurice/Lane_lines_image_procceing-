import cv2
import matplotlib.image as mpimg
import numpy as np

def hist(img):
    bottom_half = img[img.shape[0]//2:,:]
    return np.sum(bottom_half, axis=0)

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

    # a method that extracts the feature of an image and store them in the class
    def extract_features(self, my_photo):
        self.img = my_photo
        self.window_height = np.int(my_photo.shape[0]//self.nwindows)
        self.nonzero = my_photo.nonzero()
        self.nonzerox = np.array(self.nonzero[1])
        self.nonzeroy = np.array(self.nonzero[0])

    # a method that finds the lane of an image
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

    # a method that detects the lanes of a given image through taking a binary image and returning an image with the lane pixels
    def forward(self, img):
        self.extract_features(img)
        return self.fit_poly(img)

    # return all pixels in a specific window
    # inputs:
    #   center (tuple): coordinates of the center of the window
    #   margin (int): half width of the window
    #   height (int): height of the window
    # outputs:
    #   (np.array): x coordinates of pixels that lie inside the window
    #   (np.array): y coordinates of pixels that lie inside the window
    def pixels_in_window(self, center, margin, height):
        top_left = (center[0]-margin, center[1]-height//2)
        bottom_right = (center[0]+margin, center[1]+height//2)

        conditionx = (top_left[0] <= self.nonzerox) & (self.nonzerox <= bottom_right[0])
        conditiony = (top_left[1] <= self.nonzeroy) & (self.nonzeroy <= bottom_right[1])
        return self.nonzerox[conditionx&conditiony], self.nonzeroy[conditionx&conditiony]


    # return lane pixels from a binary warped image
    # inputs:
    #   img (np.array): a binary warped image
    # outputs:
    #   left_x (np.array): x coordinates of left lane pixels
    #   left_y (np.array): y coordinates of left lane pixels
    #   right_x (np.array): x coordinates of right lane pixels
    #   right_y (np.array): y coordinates of right lane pixels
    #   output_img (np.array): A RGB image that use to display result later on
    def find_lane_pixels(self, img):
        assert(len(img.shape) == 2)

        output_img = np.dstack((img, img, img))

        histogram = hist(img)
        middle_point = histogram.shape[0]//2
        left_x_base = np.argmax(histogram[:middle_point])
        right_x_base = np.argmax(histogram[middle_point:]) + middle_point

        left_x_current = left_x_base
        right_x_current = right_x_base
        y_current = img.shape[0] + self.window_height//2

        left_x, left_y, right_x, right_y = [], [], [], []

        for _ in range(self.nwindows):
            y_current -= self.window_height
            center_left = (left_x_current, y_current)
            center_right = (right_x_current, y_current)

            good_left_x, good_left_y = self.pixels_in_window(center_left, self.margin, self.window_height)
            good_right_x, good_right_y = self.pixels_in_window(center_right, self.margin, self.window_height)

            left_x.extend(good_left_x)
            left_y.extend(good_left_y)
            right_x.extend(good_right_x)
            right_y.extend(good_right_y)

            if len(good_left_x) > self.minpix:
                left_x_current = np.int32(np.mean(good_left_x))
            if len(good_right_x) > self.minpix:
                right_x_current = np.int32(np.mean(good_right_x))

        return left_x, left_y, right_x, right_y, output_img
    def plot(self, out_img):
        np.set_printoptions(precision=6, suppress=True)
        lR, rR, pos = self.measure_curvature()

        value = None
        if abs(self.left_fit[0]) > abs(self.right_fit[0]):
            value = self.left_fit[0]
        else:
            value = self.right_fit[0]

        if abs(value) <= 0.00015:
            self.dir.append('F')
        elif value < 0:
            self.dir.append('L')
        else:
            self.dir.append('R')
        
        if len(self.dir) > 10:
            self.dir.pop(0)

        W = 400
        H = 500
        widget = np.copy(out_img[:H, :W])
        widget //= 2
        widget[0,:] = [0, 0, 255]
        widget[-1,:] = [0, 0, 255]
        widget[:,0] = [0, 0, 255]
        widget[:,-1] = [0, 0, 255]
        out_img[:H, :W] = widget

        direction = max(set(self.dir), key = self.dir.count)
        msg = "Keep Straight Ahead"
        curvature_msg = "Curvature = {:.0f} m".format(min(lR, rR))
        if direction == 'L':
            y, x = self.left_curve_img[:,:,3].nonzero()
            out_img[y, x-100+W//2] = self.left_curve_img[y, x, :3]
            msg = "Left Curve Ahead"
        if direction == 'R':
            y, x = self.right_curve_img[:,:,3].nonzero()
            out_img[y, x-100+W//2] = self.right_curve_img[y, x, :3]
            msg = "Right Curve Ahead"
        if direction == 'F':
            y, x = self.keep_straight_img[:,:,3].nonzero()
            out_img[y, x-100+W//2] = self.keep_straight_img[y, x, :3]

        cv2.putText(out_img, msg, org=(10, 240), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        if direction in 'LR':
            cv2.putText(out_img, curvature_msg, org=(10, 280), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

        cv2.putText(
            out_img,
            "Good Lane Keeping",
            org=(10, 400),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.2,
            color=(0, 255, 0),
            thickness=2)

        cv2.putText(
            out_img,
            "Vehicle is {:.2f} m away from center".format(pos),
            org=(10, 450),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.66,
            color=(255, 255, 255),
            thickness=2)

        return out_img

    def measure_curvature(self):
        ym = 30/720
        xm = 3.7/700

        left_fit = self.left_fit.copy()
        right_fit = self.right_fit.copy()
        y_eval = 700 * ym

        # Compute R_curve (radius of curvature)
        left_curveR =  ((1 + (2*left_fit[0] *y_eval + left_fit[1])**2)**1.5)  / np.absolute(2*left_fit[0])
        right_curveR = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

        xl = np.dot(self.left_fit, [700**2, 700, 1])
        xr = np.dot(self.right_fit, [700**2, 700, 1])
        pos = (1280//2 - (xl+xr)//2)*xm
        return left_curveR, right_curveR, pos 
    