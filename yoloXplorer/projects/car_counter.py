import math

import numpy as np

from sort import *
import torch
import cvzone
import cv2
from ultralytics import YOLO

if torch.cuda.is_available():
    print("running on GPU")
else:
    print("running on CPU")

# Computer Vision YOLO Model
model = YOLO('../weights/yolov8n.pt')

# Object names
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed",
    "dining table", "toilet", "tv monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

# Read the mask for entry cars
entry_mask = cv2.imread("./masks/cars_entry_mask.png")
print("Entry Mask :",entry_mask.shape)

# Read the mask for exit cars
exit_mask = cv2.imread("./masks/cars_exit_mask.png")
print("Exit mask :", exit_mask.shape)

# Entry Line to count the entry cars
entry_limit = [0, 315, 400, 315]

# To Track Objects and aviod duplication while counting we will use the sort
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Reading video
rec = cv2.VideoCapture('../core/videos/cars.mp4')

# Entry car counts
entry_list = []

while rec.isOpened():
    # read the video frame by frame
    ret, frame = rec.read()

    # masking using bitwise AND operation
    entry_frame = cv2.bitwise_and(frame, entry_mask)

    # Tracking Detections - Start with empty
    detections = np.empty((0,5))

    # Track the objects and count the cars
    results = model(entry_frame, stream = True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get the object positions
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = (x2-x1), (y2-y1)

            # take confidence percent and Object name
            conf = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Detect only Car
            if currentClass == 'car' and conf > 0.3:

                # corner box
                # cvzone.cornerRect(frame, bbox=(x1, y1, w, h), l=15, t=2, rt= 5)

                # showing object name with confidence
                # cvzone.putTextRect(frame, f"{currentClass}{conf}", (max(0,x1), max(35, y1)),
                #                    scale=1, thickness=2, offset = 5)

                # Array format must be [x1,y1,x2,y2,conf]
                currentArray = np.array([x1,y1,x2,y2,conf])

                # Stack the current detection array in tracking detections
                detections = np.vstack((detections, currentArray))

    # pass the total tracking detections
    resultTrackers = tracker.update(detections)

    # Line to count the entry cars
    cv2.line(frame, (entry_limit[0], entry_limit[1]), (entry_limit[2], entry_limit[3]), color=(0, 0, 255), thickness=5)

    for track_result in resultTrackers:
        x1, y1, x2, y2, Id = track_result
        x1, y1, x2, y2, Id = int(x1), int(y1), int(x2), int(y2), int(Id)
        print(track_result)
        w, h = (x2 - x1), (y2 - y1)

        # corner box with red border
        cvzone.cornerRect(frame, bbox=(x1, y1, w, h), l=15, t=2, colorR=(255,0,255))

        # showing Trading ID
        cvzone.putTextRect(frame, f"{Id}", (max(0,x1), max(35, y1)),
                           scale=1, thickness=2, offset = 10)

        # Counting cars using center points of the border if the car crosses the red line
        cx, cy =  x1+w // 2 , y1+h // 2
        cv2.circle(frame, (cx,cy), 5,(255,0,255), -1)

        if entry_limit[0]<cx<entry_limit[2] and (entry_limit[1]-35)<cy<(entry_limit[3] + 35):
            if entry_list.count(Id) == 0:
                entry_list.append(Id)

    # Showing
    cvzone.putTextRect(frame, f'Entry Count : {len(entry_list)}', (50, 50))

    cv2.imshow("Car Counter", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

rec.release()
cv2.destroyAllWindows()
