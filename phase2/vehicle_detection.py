import matplotlib.image as mpimg #
import matplotlib.pyplot as plt #
import numpy as np #
import cv2 #
from phase2.bounding_boxes import BoundingBoxes #
from moviepy.editor import VideoFileClip #
from phase2.heatmap import apply_threshold, heatmap_from_detections, get_labeled_bboxes
from phase2.detection_utils import find_cars_multiscale
from scipy.ndimage import label


# Detects vehicles in the frame.
def detect_vehicles(image, y_start, y_stop, scale, cells_per_step, avgBoxes=None, thresh=1, useHeatmap=True, debug=False, svc_data=None):
    
    sliding_window_output = find_cars_multiscale(image, svc_data, y_start, y_stop, scale, cells_per_step)
    
    if avgBoxes:
        avgBoxes.add(sliding_window_output)
        sliding_window_output = avgBoxes.all_boxes
    
    heatmap=[]
    labels=[]
        
    if useHeatmap:
        heatmap = heatmap_from_detections(image, sliding_window_output)
        heatmap_thresholded = apply_threshold(heatmap, thresh)
        
        labels = label(heatmap_thresholded)
        bboxes = get_labeled_bboxes(labels)
    else:
        bboxes = sliding_window_output
    
    if debug == False:
        return bboxes
    else:    
        return bboxes, sliding_window_output, heatmap, labels


# Draws bounding boxes on the image
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy


# Draw debug mode output (output of sliding window, heatmap, labels)
def draw_debug_board(img, bboxes, sliding_window_output, heatmap, labels):
    
    # prepare RGB heatmap image from float32 heatmap channel
    img_heatmap = (np.copy(heatmap) / np.max(heatmap) * 255.).astype(np.uint8)
    img_heatmap = cv2.applyColorMap(img_heatmap, colormap=cv2.COLORMAP_HOT)
    img_heatmap = cv2.cvtColor(img_heatmap, cv2.COLOR_BGR2RGB)

    # prepare RGB labels image from float32 labels channel
    img_labels = (np.copy(labels) / np.max(labels) * 255.).astype(np.uint8)
    img_labels = cv2.applyColorMap(img_labels, colormap=cv2.COLORMAP_HOT)
    img_labels = cv2.cvtColor(img_labels, cv2.COLOR_BGR2RGB)
    
    # draw sliding_window_output in the frame
    img_sliding_window_output = np.copy(img)
    img_sliding_window_output = draw_boxes(img_sliding_window_output, sliding_window_output, thick=2)
    
    # setting coordinates for the debug board
    ymax = 0
    
    board_x = 5
    board_y = 5
    board_ratio = (img.shape[0] - 3*board_x)//3 / img.shape[0]
    board_h = int(img.shape[0] * board_ratio)
    board_w = int(img.shape[1] * board_ratio)
        
    ymin = board_y
    ymax = board_h + board_y
    xmin = board_x
    xmax = board_x + board_w

    offset_x = board_x + board_w

    # resizing sliding_window_output
    img_sliding_window_output = cv2.resize(img_sliding_window_output, dsize=(board_w, board_h), interpolation=cv2.INTER_LINEAR)
    img[ymin:ymax, xmin:xmax, :] = img_sliding_window_output
    
    # resizing heatmap
    xmin += offset_x
    xmax += offset_x
    img_heatmap = cv2.resize(img_heatmap, dsize=(board_w, board_h), interpolation=cv2.INTER_LINEAR)
    img[ymin:ymax, xmin:xmax, :] = img_heatmap
    
    # resizing lables
    xmin += offset_x
    xmax += offset_x
    img_labels = cv2.resize(img_labels, dsize=(board_w, board_h), interpolation=cv2.INTER_LINEAR)
    img[ymin:ymax, xmin:xmax, :] = img_labels
    
    return img



# Detects and lables vehicles in the frame
def pipeline(img, 
    y_start=[405, 400, 500], 
    y_stop=[510, 600, 710], 
    scale=[1, 1.5, 2.5], 
    cells_per_step=2, 
    thresh=1,
    useHeatmap=True, avgBoxes=None, debug=False, svc_data=None):
    
    assert(svc_data)
    
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255

    if debug == False:
        bboxes = detect_vehicles(img, y_start, y_stop, scale, cells_per_step, thresh=thresh, useHeatmap=useHeatmap, avgBoxes=avgBoxes, svc_data=svc_data)
        result = draw_boxes(draw_img, bboxes, thick=3)
    else:
        bboxes, hot_windows, heatmap, labels = detect_vehicles(img, y_start, y_stop, scale, cells_per_step, thresh=thresh, useHeatmap=useHeatmap, avgBoxes=avgBoxes, debug=True, svc_data=svc_data)
        result = draw_boxes(draw_img, bboxes, thick=3)    
        result = draw_debug_board(result, bboxes, hot_windows, heatmap, labels[0])

    return result
