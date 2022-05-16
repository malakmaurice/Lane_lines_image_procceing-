import numpy as np

# Returns heatmap for list of bounding boxes
def heatmap_from_detections(img, bbox_list):
    h,w,_ = img.shape
    heatmap = np.zeros((h,w)).astype(np.float32)

    for box in bbox_list:
        x1,y1 = box[0]
        x2,y2 = box[1]
        heatmap[y1:y2, x1:x2] += 1

    return heatmap
       
# Apply a threshold to heatmap
def apply_threshold(heatmap, threshold):
    heatmap = np.copy(heatmap)
    heatmap[heatmap <= threshold] = 0
    heatmap = np.clip(heatmap, 0, 255)
    
    return heatmap


# Returns list of bounding boxes for detected labes (cars)
def get_labeled_bboxes(labels):
    bboxes = []

    for id in range(1, labels[1]+1):
        nonzero = (labels[0] == id).nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])
        bbox = ((np.min(nonzero_x), np.min(nonzero_y)), (np.max(nonzero_x), np.max(nonzero_y)))        
        bboxes.append(bbox)
    
    return bboxes