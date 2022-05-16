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
from phase2.train import train
from phase2.save_load_pickle import load_classifier
from phase2.vehicle_detection import pipeline
from phase2.bounding_boxes import BoundingBoxes
import time

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
        
        added_img = np.concatenate((img5,img4),axis = 1)
     
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


class DetectCars:
    def __init__(self, classifier_path):
        self.svc_data = load_classifier(classifier_path)
    
    # takes an image variable
    def process_frame_debug(self, img):
        avgBoxes = BoundingBoxes()
        DEBUG = True
        result = pipeline(img, avgBoxes=avgBoxes, debug=DEBUG, svc_data=self.svc_data)      
        return result

    # takes an image variable
    def process_frame(self, img):
        avgBoxes = BoundingBoxes()
        DEBUG = False
        result = pipeline(img, avgBoxes=avgBoxes, debug=DEBUG, svc_data=self.svc_data)      
        return result

    # takes image path
    def process_image(self, input_path, output_path):
        img = mpimg.imread(input_path)
        t1 = time.time()
        result = self.process_frame(img)
        t2 = time.time()
        mpimg.imsave(output_path, result)
        print("Time taken: ", round(t2-t1,2), " seconds")

    # takes image path
    def process_image_debug(self, input_path, output_path):
        img = mpimg.imread(input_path)
        t1 = time.time()
        result = self.process_frame_debug(img)
        t2 = time.time()
        mpimg.imsave(output_path, result)
        print("Time taken: ", round(t2-t1,2), " seconds")

    # takes video path
    def process_video(self, input_path, output_path):
        clip = VideoFileClip(input_path)
        out_clip = clip.fl_image(self.process_frame) 
        out_clip.write_videofile(output_path, audio=False)

    # takes video path
    def process_video_debug(self, input_path, output_path):
        clip = VideoFileClip(input_path)
        out_clip = clip.fl_image(self.process_frame_debug) 
        out_clip.write_videofile(output_path, audio=False)





def main():

    # For Phase 1
    findLaneLines = FindLaneLines()

    # For Phase 2
    classifier_path = "phase2/Classifier.p"
    data_path = "phase2/data"

    operation = sys.argv[1] 
    type = sys.argv[2] if operation != "--train-model" else None
    debug = sys.argv[3] if operation != "--train-model" else None
    input = sys.argv[4] if operation != "--train-model" else None
    output = sys.argv[5] if operation != "--train-model" else None
   
    if operation == "--detect-lane":
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
    
    elif operation == "--detect-cars":
        if type == "--image":
            if debug == "--no-debug":
                detectCars = DetectCars(classifier_path)
                detectCars.process_image(input, output)
            elif debug == "--debug":
                detectCars = DetectCars(classifier_path)
                detectCars.process_image_debug(input, output)
            else:
                print("Unsupported Mode")
        elif type == "--video":
            if debug == "--no-debug":
                detectCars = DetectCars(classifier_path)
                detectCars.process_video(input, output)
            elif debug == "--debug":
                detectCars = DetectCars(classifier_path)
                detectCars.process_video_debug(input, output)
            else:
                print("Unsupported Mode")   
        else:
            print("Unsupported Operation")
    
    elif operation == "--train-model":
        train(data_path, classifier_path, debug=True)


    

    # detectCars = DetectCars(classifier_path)
    # detectCars.process_image_debug("test_images/test4.jpg", "test_images/tt.jpg")




if __name__ == "__main__":
    main()