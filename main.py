import sys
from cv2 import resize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import sys
from IPython.display import HTML, Video
from moviepy.editor import VideoFileClip
from CameraCalibration import CameraCalibration
from Thresholding import *
from PerspectiveTransformation import *
from Lanelines import *

class FindLaneLines:
    """ This class is for parameter tunning.

    Attributes:
        ...
    """
    def __init__(self):
        """ Init Application"""
        self.calibration = CameraCalibration('camera_cal', 9, 6)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()
    def resize_image(self,img,ratio):
        img=cv2.resize(img,(int(img.shape[1]*ratio),int(img.shape[0]*ratio)))
        return img

    def forward_without_debug(self,img):
        out_img = np.copy(img)
        img = self.calibration.undistort(img)
        img = self.transform.forward(img)
        img = self.thresholding.forward(img)
        img = self.lanelines.forward(img)
        img = self.transform.backward(img)

        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
        out_img = self.lanelines.plot(out_img)
        return out_img

    def forward(self, img):
        out_img = np.copy(img)
        img = self.calibration.undistort(img)
        img1= self.resize_image(img,0.5)
        img = self.transform.forward(img)
        img2= self.resize_image(img,0.5)
        img1=np.concatenate((img1,img2),axis = 0)

        img = self.thresholding.forward(img)
        img3= self.resize_image(img,0.5)
        img3= np.dstack((img3,img3,img3))
        img1=np.concatenate((img1,img3),axis = 0)

        img = self.lanelines.forward(img)
        img4= self.resize_image(img,0.5)

        
       
        
        img = self.transform.backward(img)
        img5= self.resize_image(img,0.5)
        
        added_img = np.concatenate((img4,img5),axis = 1)
     
        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
        out_img = self.lanelines.plot(out_img)
        
        
        
        out_img=np.concatenate((out_img,added_img),axis = 0)
        out_img=np.concatenate((out_img,img1),axis = 1)
        
        return out_img

    def process_image(self, input_path, output_path):
        img = mpimg.imread(input_path)
        out_img = self.forward_without_debug(img)
        mpimg.imsave(output_path, out_img)

    def process_image_debug(self, input_path, output_path):
        img = mpimg.imread(input_path)
        out_img = self.forward(img)
        mpimg.imsave(output_path, out_img)

    def process_video(self, input_path, output_path,):

        clip = VideoFileClip(input_path)
        out_clip = clip.fl_image(self.forward_without_debug)
        out_clip.write_videofile(output_path, audio=False)
        
    def process_video_debug(self, input_path, output_path):
        clip = VideoFileClip(input_path)
        out_clip = clip.fl_image(self.forward)
        out_clip.write_videofile(output_path, audio=False)

def main():

    # For Phase 1
    findLaneLines = FindLaneLines()
    type = sys.argv[1]
    debug = sys.argv[2]
    input = sys.argv[3]
    output = sys.argv[4]
    
    if type == "--image":
        if debug == "--no-debug":
            findLaneLines.process_image(input, output)
        elif debug == "--debug":
            findLaneLines.process_image_debug(input, output)
        else:
            print("Unsupported Mode")
    elif type == "--video":
        if debug == "--no-debug":
            findLaneLines.process_video(input, output)
        elif debug == "--debug":
            findLaneLines.process_video_debug(input, output)
        else:
            print("Unsupported Mode")   
    else:
        print("Unsupported Operation")


if __name__ == "__main__":
    main()