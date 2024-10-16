import cv2
import numpy as np
import math
from ultralytics import YOLO

# Load a COCO-pretrained YOLO11x model
model = YOLO("yolo11x.pt")

# Define HSV ranges for a wide variety of colors
COLOR_RANGES = {
    "red": [(0, 50, 50), (10, 255, 255), (160, 50, 50), (180, 255, 255)],  # Red has two ranges
    "green": [(36, 50, 50), (89, 255, 255)],
    "blue": [(90, 50, 50), (130, 255, 255)],
    "yellow": [(15, 50, 50), (35, 255, 255)],
    "orange": [(10, 100, 100), (25, 255, 255)],
    "purple": [(130, 50, 50), (160, 255, 255)],
    "pink": [(160, 50, 50), (170, 255, 255)],  # Pink is between red and purple
    "cyan": [(80, 50, 50), (100, 255, 255)],
    "brown": [(10, 100, 20), (20, 255, 200)],  # Brown is close to dark orange/yellow
    "gray": [(0, 0, 50), (180, 50, 200)],      # Gray has low saturation values
    "lime": [(36, 100, 100), (70, 255, 255)],  # Bright green/lime
    "olive": [(22, 50, 50), (35, 255, 180)],   # Dark yellowish-green
    "teal": [(70, 50, 50), (90, 255, 255)],    # Mix of blue and green
    "navy": [(100, 50, 50), (130, 255, 180)],  # Dark blue
    "magenta": [(140, 50, 50), (160, 255, 255)],  # Bright pink-purple
    "beige": [(20, 50, 50), (35, 255, 255)],   # Light yellow-brown
    "maroon": [(0, 50, 50), (10, 100, 100)],   # Dark red
    "violet": [(130, 50, 50), (160, 255, 255)],  # Purple with red tint
    "gold": [(20, 100, 100), (30, 255, 255)],  # Golden yellow
    "silver": [(0, 0, 50), (180, 50, 150)],    # Low saturation and brightness for metallic silver
    "indigo": [(111, 50, 50), (130, 255, 255)],  # Deep blue with purple tint
    "turquoise": [(80, 100, 100), (100, 255, 255)],  # Blue-green mix
}

def detect_dominant_color(bbox, image):
    # Extract the bounding box region of interest (ROI)
    x, y, w, h = bbox
    # roi = image[y:y+h, x:x+w]
    roi = image[int(y):int(y + h), int(x):int(x + w)]

    # Convert the ROI to the HSV color space
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    color_percentages = {}

    # Loop over each color range and compute the percentage of that color in the ROI
    for color_name, ranges in COLOR_RANGES.items():
        # Combine masks for ranges that have more than one HSV range (like red)
        mask = None
        for (lower, upper) in zip(ranges[::2], ranges[1::2]):
            lower_bound = np.array(lower)
            upper_bound = np.array(upper)
            color_mask = cv2.inRange(hsv_roi, lower_bound, upper_bound)

            # Combine multiple ranges into a single mask
            if mask is None:
                mask = color_mask
            else:
                mask = cv2.bitwise_or(mask, color_mask)

        # Calculate the percentage of the color in the bounding box
        color_ratio = cv2.countNonZero(mask) / (w * h)
        color_percentages[color_name] = color_ratio

    # Find the dominant color (the one with the highest percentage)
    dominant_color = max(color_percentages, key=color_percentages.get)
    dominant_color_percentage = color_percentages[dominant_color]

    # Return the dominant color and its percentage
    return dominant_color, dominant_color_percentage

def detect_objects(image_path):
    '''

    :param image_path:
    :return: tuple of
    names
    confidences
    colors of detected objects
    percentages of colors
    summary of detected objects
    '''
    names = []
    confidences = []
    colors = []
    color_percentages = []

    # Count occurrences of each class
    object_counts = {}

    results = model(image_path)

    for r in results:
        for box in r.boxes:
            confidence = math.ceil((box.conf[0] * 100)) / 100
            # x1, y1, x2, y2 = box.xyxy[0]

            # Detect color
            image = cv2.imread(image_path)
            detected_color, color_percentage = detect_dominant_color(box.xyxy[0], image)

            idxClassName = int(box.cls[0])
            className = r.names[idxClassName]
            print(className, ' ', confidence)

            names.append(className)
            confidences.append(confidence)

            colors.append(detected_color)
            color_percentages.append(color_percentage)

            if className in object_counts:
                object_counts[className] += 1
            else:
                object_counts[className] = 1

    # Generate object count summary (e.g., "Number of persons: 5")
    object_summary = ', '.join([f"Number of {cls}: {count}" for cls, count in object_counts.items()])

    return names, confidences, colors, color_percentages, object_counts

# Loop through keyframes and detect objects
def process_keyframes_for_objects(keyframes):
    video_objects = {}
    for video, frames in keyframes.items():
        video_objects[video] = {}
        for frame in frames:
            objects, _ = detect_objects(frame)
            video_objects[video][frame] = objects
    return video_objects