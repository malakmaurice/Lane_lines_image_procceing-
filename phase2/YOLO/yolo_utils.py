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



# get the layer names from the network
def get_layer_names(net):
    names = net.getLayerNames()
    return [names[i- 1] for i in net.getUnconnectedOutLayers()]

# creates a blob from the image to be used in the neural network
def create_blob_from_image(img, scale_factor, size, crop, swap_red_and_blue):
    return cv2.dnn.blobFromImage(img, scale_factor, size, crop=crop, swapRB=swap_red_and_blue)

# applies non maximum suppression to the bounding boxes
def apply_nms(bboxes, confidences, score_thresh, nms_thresh):
    return cv2.dnn.NMSBoxes(bboxes, confidences, score_thresh, nms_thresh)



# draw the bounding boxes and writes the labels on the image
def draw_boxes_with_labels(img, idxs, bboxes, confidences, classIDs, labels, font_scale, thick):
    draw_img = np.copy(img)

    for i in idxs.flatten():
        (x,y) = [bboxes[i][0], bboxes[i][1]]
        (w,h) = [bboxes[i][2], bboxes[i][3]]
    
        label_name = labels[classIDs[i]]
        label_confidence = confidences[i]
        message = label_name + ": " + str(label_confidence)

        cv2.rectangle(draw_img, (x,y), (x+w,y+h), (0,255,0), thick)
        cv2.putText(draw_img, message, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,0), 2)

    return draw_img