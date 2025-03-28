from ultralytics import YOLO
import  cv2

# YOLO model
# yolov8n - Nano   - fast and less accurate
# yolov8m - Medium - Medium pace and normal accuracy
# yolov8l - Large  - slow and accurate

model = YOLO('../weights/yolov8m.pt')

# Pass the image to the model and show the result
results = model('./images/cars_highway.jpg', show = True)

# Wait key is show the result until we close
cv2.waitKey(0)