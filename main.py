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
from phase2.YOLO.yolo_utils import *

# Detect Lane Lines
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

# Detect Cars Using YOLO
class DetectCarsYOLO:
    def __init__(self, weights_path, config_path, labels_path):
        self.net = create_net(weights_path, config_path)
        self.labels = get_labels(labels_path)
    
    def process_frame(self, img):
        (height, width, _) = img.shape

        blob = create_blob_from_image(img=img, scale_factor=1/255, size=(416, 416), crop=False, swap_red_and_blue=False)
        self.net.setInput(blob)

        layers_names = get_layer_names(net=self.net)
        layers_output = self.net.forward(layers_names)

        bboxes, confidences, classIDs = analyze_output(layers_output=layers_output, img_height=height, img_width=width, confidence_thresh=0.85)

        idx = apply_nms(bboxes=bboxes, confidences=confidences, score_thresh=0.8, nms_thresh=0.8)

        final_img = draw_boxes_with_labels(img=img, idxs=idx, bboxes=bboxes, confidences=confidences, classIDs=classIDs, labels=self.labels, font_scale=0.5, thick=3)

        return final_img
    
    def process_image(self, input_path, output_path):
        img = mpimg.imread(input_path)
        t1 = time.time()
        result = self.process_frame(img)
        t2 = time.time()
        mpimg.imsave(output_path, result)
        print("YOLO Time taken: ", round(t2-t1,2), " seconds")

    def process_video(self, input_path, output_path):
        clip = VideoFileClip(input_path)
        out_clip = clip.fl_image(self.process_frame)
        out_clip.write_videofile(output_path, audio=False)

# Detect Cars Using SVM Model
class DetectCarsSVM:
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
        print("SVM Time taken: ", round(t2-t1,2), " seconds")

    # takes image path
    def process_image_debug(self, input_path, output_path):
        img = mpimg.imread(input_path)
        t1 = time.time()
        result = self.process_frame_debug(img)
        t2 = time.time()
        mpimg.imsave(output_path, result)
        print("SVM Time taken: ", round(t2-t1,2), " seconds")

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

    # Entry argument
    operation = sys.argv[1]

    # Declare arguments
    data_path = None
    type = None
    debug = None
    input = None
    output = None
    weights_path = None
    config_path = None
    labels_path = None


    # Obtain required arguments for each operation

    if operation == "--train-svm-model":
        data_path = sys.argv[2]
    
    elif (operation == "--detect-lane" or operation == "--detect-cars-svm"):
        type = sys.argv[2]
        debug = sys.argv[3]
        input = sys.argv[4]
        output = sys.argv[5]
    
    elif operation == "--detect-cars-yolo":
        weights_path = sys.argv[2]
        config_path = sys.argv[3]
        labels_path = sys.argv[4]
        type = sys.argv[5]
        input = sys.argv[6]
        output = sys.argv[7]


    # Apply checks on arguments

    if operation != "--train-svm-model" and operation != "--detect-lane" and operation != "--detect-cars-svm" and operation != "--detect-cars-yolo":
        raise Exception("Invalid Argument for Operation\nPossible Options:\n--train-svm-model\n--detect-lane\n--detect-cars-svm\n--detect-cars-yolo")

    if type != None and type != "--image" and type != "--video":
        print(type)
        raise Exception("Invalid Argument for Type\nPossible Options:\n--image\n--video")
    
    if debug != None and debug != "--debug" and debug != "--no-debug":
        raise Exception("Invalid Argument for Debug\nPossible Options:\n--debug\n--no-debug")
    
    if operation == "--train-model" and data_path == None:
        raise Exception("Invalid Argument for Data Path. Please provide a valid path")

   
    # Perform operations

    if operation == "--detect-lane":
        if type == "--image":
            if debug == "--no-debug":
                findLaneLines.process_image(input, output)
            elif debug == "--debug":
                findLaneLines.process_image_debug(input, output)
        elif type == "--video":
            if debug == "--no-debug":
                findLaneLines.process_video(input, output)
            elif debug == "--debug":
                findLaneLines.process_video_debug(input, output) 
    
    elif operation == "--detect-cars-svm":
        if type == "--image":
            if debug == "--no-debug":
                detect_svm = DetectCarsSVM(classifier_path)
                detect_svm.process_image(input, output)
            elif debug == "--debug":
                detect_svm = DetectCarsSVM(classifier_path)
                detect_svm.process_image_debug(input, output)
        elif type == "--video":
            if debug == "--no-debug":
                detect_svm = DetectCarsSVM(classifier_path)
                detect_svm.process_video(input, output)
            elif debug == "--debug":
                detect_svm = DetectCarsSVM(classifier_path)
                detect_svm.process_video_debug(input, output)
    
    elif operation == "--detect-cars-yolo":
        if type == "--image":
            detect_yolo = DetectCarsYOLO(weights_path, config_path, labels_path)
            detect_yolo.process_image(input, output)
        elif type == "--video":
            detect_yolo = DetectCarsYOLO(weights_path, config_path, labels_path)
            detect_yolo.process_video(input, output)

    elif operation == "--train-svm-model":
        train(data_path, classifier_path, debug=True)



if __name__ == "__main__":
    main()