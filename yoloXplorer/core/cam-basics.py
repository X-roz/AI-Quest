from ultralytics import YOLO
# import time
import cv2

model = YOLO('../weights/yolov8n.pt')
cam = cv2.VideoCapture(0)

# setting size  to show the captured video
cam.set(3,640)
cam.set(4, 480)

while cam.isOpened():
    ret, frame = cam.read()
    result = model(frame,show = True)
    # cv2.imshow("Welcome", result)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
