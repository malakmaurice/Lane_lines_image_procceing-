from collections import deque

# The class is used to store the bounding boxes of the detected objects for the last n frames
class BoundingBoxes:
    def __init__(self, number_of_frames = 5):
        self.number_of_frames = number_of_frames
        self.recent_boxes = deque([], maxlen=number_of_frames)
        self.current_boxes = None
        self.all_boxes = []
    
    def update_all_boxes_(self):
        all_boxes = []
        for boxes in self.recent_boxes:
            all_boxes += boxes
        if len(all_boxes) == 0:
            self.all_boxes = []
        else:
            self.all_boxes = all_boxes
    
    def add(self, boxes):
        self.current_boxes = boxes
        self.recent_boxes.appendleft(boxes)
        self.update_all_boxes_()