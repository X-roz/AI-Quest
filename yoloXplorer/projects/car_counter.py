import math

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

# Reading video
rec = cv2.VideoCapture('../core/videos/cars.mp4')

while rec.isOpened():
    # read the video frame by frame
    ret, frame = rec.read()

    # masking using bitwise AND operation
    entry_frame = cv2.bitwise_and(frame, entry_mask)

    # track the objects and count the cars
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
                cvzone.cornerRect(frame, bbox=(x1, y1, w, h), l=15, t=2)

                # showing object name with confidence
                cvzone.putTextRect(frame, f"{currentClass}{conf}", (max(0,x1), max(35, y1)),
                                   scale=1, thickness=2, offset = 5)

    cv2.imshow("Car Counter", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

rec.release()
cv2.destroyAllWindows()
