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

# Reading video
rec = cv2.VideoCapture('../core/videos/cars.mp4')

while rec.isOpened():
    # read the video frame by frame
    ret, frame = rec.read()

    # track the objects and count the cars
    results = model(frame, stream = True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get the object positions
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)

            # draw boxes
            w, h = (x2-x1), (y2-y1)
            cvzone.cornerRect(frame, bbox=(x1, y1, w, h))

            # take confidence percent and Object name
            conf = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])

            # show confidence value with object name
            cvzone.putTextRect(frame, f"{classNames[cls]}{conf}", (max(0,x1), max(35, y1)), scale=1, thickness=2)

    cv2.imshow("Car Counter" , frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

rec.release()
cv2.destroyAllWindows()
