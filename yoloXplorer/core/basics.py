from ultralytics import YOLO
import  cv2

model = YOLO('../weights/yolov8n.pt')
results = model('./images/schl_bus.jpg', show = True)
cv2.waitKey(0)