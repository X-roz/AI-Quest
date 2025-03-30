from ultralytics import YOLO
import cvzone
import cv2
import math

# To Check whether teh code is running on GPU
# import torch
# if torch.cuda.is_available():
#     print("GPU is available")
# else:
#     print("GPU is not available")

model = YOLO('../weights/yolov8n.pt')

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

# cam = cv2.VideoCapture(0) # for Webcam object detection
cam = cv2.VideoCapture('./videos/cars.mp4') # video file - object detection

# setting size  to show the captured video
cam.set(3,840)
cam.set(4, 680)

while cam.isOpened():
    ret, frame = cam.read()
    result = model(frame,stream = True)
    for r in result:
        boxes = r.boxes
        for box in boxes:
           # Get the postions of the objects
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 5)

            # Draw boxes around objects
            w, h = int(x2-x1), int(y2-y1)
            cvzone.cornerRect(frame, (x1,y1,w,h))

            # Get class name of the object and its confidence level (2 decimal points)
            conf = math.ceil(box.conf[0]*100)/100
            cls = int(box.cls[0])

            # show the classname with confidence level above the object
            # Scale parameter will squeeze down everything to make it smaller relly helpful when lots of objects in the frame (default - 3)
            cvzone.putTextRect(frame, f'{classNames[cls]}{conf}', (max(0,x1), max(35,y1)), scale = 1, thickness =2)


    cv2.imshow("YOLO SAMPLE", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
