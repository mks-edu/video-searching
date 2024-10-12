# Refer: https://docs.ultralytics.com/models/yolo11/#__tabbed_2_1
import math
from ultralytics import YOLO

# Load a COCO-pretrained YOLO11x model
model = YOLO("yolo11x.pt")

# Run inference with the YOLO11n model on the 'bus.jpg' image
image_frame = r"\Keyframe\Keyframes_L01\keyframes\L01_V001\032.jpg"
results = model(image_frame)

for r in results:
    for box in r.boxes:
        confidence = math.ceil((box.conf[0] * 100)) / 100
        # x1, y1, x2, y2 = box.xyxy[0]

        idxClassName = int(box.cls[0])
        className = r.names[idxClassName]
        print(className, ' ', confidence)
