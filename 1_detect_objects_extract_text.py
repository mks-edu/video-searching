import os
import cv2
from keyframe.detect_object import detect_objects
from keyframe.text import extract_text_img
from keyframe.frame import save_keyframe_data

def process_keyframes(video_folder):
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
            detected_objects, confidences, colors, color_percentages, object_counts = detect_objects(frame_path)

            # Text Extraction (OCR)
            text = extract_text_img(img)

            keyframe_data.append({
                'keyframe': keyframe,
                'detected_objects': detected_objects,
                'colors': colors,
                'color_percentages': color_percentages,
                'object_counts': object_counts,
                'extracted_text': text,
            })

        data[video_name] = keyframe_data

    return data


# Example usage:
input_folder = r'V:\AIC-2024\Data_2024\Keyframe\Keyframes_L29\keyframes'
output_json_fpath = "1_keyframe_data_Keyframes_L29.json"
processed_data = process_keyframes(input_folder)
save_keyframe_data(processed_data, output_json_fpath)
