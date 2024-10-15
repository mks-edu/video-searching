import os
import json
import cv2
from keyframe.detect_object import detect_objects
from keyframe.text import extract_text_img

def process_keyframes(video_folder, output_json_fpath):
    '''
    :param video_folder: structure of folder of key frames such as
    video_name1:
         keyframe1.1.jpg
         keyframe1.2.jpg
         ...
    video_name2:
         keyframe2.1.jpg
         keyframe2.2.jpg
         ...
    :return:
     {"video_name1":
        {
            "keyframe":
            "detected_objects":
            detected_objects_summary:
            "extracted_text":
        }
     }
    '''
    data = {}
    i = 0
    for video_name in os.listdir(video_folder):
        video_path = os.path.join(video_folder, video_name)
        keyframe_data = []

        for keyframe in os.listdir(video_path):
            frame_path = os.path.join(video_path, keyframe)

            img = cv2.imread(frame_path)

            # Detect objects
            detected_objects, _, detected_objects_summary = detect_objects(frame_path)

            # Text Extraction (OCR)
            text = extract_text_img(img)

            keyframe_data.append({
                'keyframe': keyframe,
                'detected_objects': detected_objects,
                'detected_objects_summary': detected_objects_summary,
                'extracted_text': text,
            })

        data[video_name] = keyframe_data

    # Save
    with open(output_json_fpath, 'w') as f:
        json.dump(data, f)

    return data

# Example usage:
input_folder = r'V:\AIC-2024\Data_2024\Keyframe\Keyframes_L01\keyframes'
output_json_fpath = "1_keyframe_data.json"
processed_data = process_keyframes(input_folder, output_json_fpath)
