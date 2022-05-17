import numpy as np
import cv2

# Initialize a neural network using the weights and configuration files
def create_net(weights_path, config_path):
    return cv2.dnn.readNetFromDarknet(config_path, weights_path)

# Get the labels
def get_labels(labels_path):
    return open(labels_path).read().strip().split("\n")

# process the output of the neural network to get the bounding boxes, the confidences and the class ids
def analyze_output(layers_output, img_height, img_width, confidence_thresh):
    bboxes = []
    confidences = []
    classIDs = []

    for output in layers_output:
        for detection in output:
            scores= detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if confidence > confidence_thresh:
                bbox = detection[:4] * np.array([img_width, img_height, img_width, img_height])
                bx, by, bw, bh = bbox.astype("int")
                         
                x = int(bx-(bw/2))
                y = int(by-(bh/2))
                
                bboxes.append([x, y, int(bw), int(bh)])
                confidences.append(confidence)
                classIDs.append(classID)
    
    return bboxes, confidences, classIDs