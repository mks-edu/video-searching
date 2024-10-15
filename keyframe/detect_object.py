import math
from ultralytics import YOLO

# Load a COCO-pretrained YOLO11x model
model = YOLO("yolo11x.pt")

def detect_objects(image_path):
    '''

    :param image_path:
    :return: tuple of names and confidences
    '''
    names = []
    confidences = []

    results = model(image_path)

    for r in results:
        for box in r.boxes:
            confidence = math.ceil((box.conf[0] * 100)) / 100
            x1, y1, x2, y2 = box.xyxy[0]

            idxClassName = int(box.cls[0])
            className = r.names[idxClassName]
            print(className, ' ', confidence)
            names.append(className)
            confidences.append(confidence)
            #print(results)
    return names, confidences

# Loop through keyframes and detect objects
def process_keyframes_for_objects(keyframes):
    video_objects = {}
    for video, frames in keyframes.items():
        video_objects[video] = {}
        for frame in frames:
            objects, _ = detect_objects(frame)
            video_objects[video][frame] = objects
    return video_objects