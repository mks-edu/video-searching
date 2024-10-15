import math
from ultralytics import YOLO

# Load a COCO-pretrained YOLO11x model
model = YOLO("yolo11x.pt")

def detect_objects(image_path):
    '''

    :param image_path:
    :return: tuple of
    names
    confidences
    summary of detected objects
    '''
    names = []
    confidences = []

    # Count occurrences of each class
    object_counts = {}

    results = model(image_path)

    for r in results:
        for box in r.boxes:
            confidence = math.ceil((box.conf[0] * 100)) / 100
            # x1, y1, x2, y2 = box.xyxy[0]

            idxClassName = int(box.cls[0])
            className = r.names[idxClassName]
            print(className, ' ', confidence)
            names.append(className)
            confidences.append(confidence)

            if className in object_counts:
                object_counts[className] += 1
            else:
                object_counts[className] = 1

    # Generate object count summary (e.g., "Number of persons: 5")
    object_summary = ', '.join([f"Number of {cls}: {count}" for cls, count in object_counts.items()])

    return names, confidences, object_summary

# Loop through keyframes and detect objects
def process_keyframes_for_objects(keyframes):
    video_objects = {}
    for video, frames in keyframes.items():
        video_objects[video] = {}
        for frame in frames:
            objects, _ = detect_objects(frame)
            video_objects[video][frame] = objects
    return video_objects