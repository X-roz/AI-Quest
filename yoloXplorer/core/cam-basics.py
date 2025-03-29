from ultralytics import YOLO
import cvzone
import cv2

model = YOLO('../weights/yolov8n.pt')
cam = cv2.VideoCapture(0)

# setting size  to show the captured video
cam.set(3,640)
cam.set(4, 480)

while cam.isOpened():
    ret, frame = cam.read()
    result = model(frame,stream = True)
    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 5)

            w, h = int(x2-x1), int(y2-y1)
            cvzone.cornerRect(frame, (x1,y1,w,h))

    cv2.imshow("YOLO SAMPLE", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
