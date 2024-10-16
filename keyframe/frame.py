import os
import json

# Function to load all images from subfolders
def load_keyframes(input_folder):
    '''

    :param input_folder:
    :return: map[video] of keyframes
    '''
    keyframes = {}
    for video in os.listdir(input_folder):
        video_folder = os.path.join(input_folder, video)
        if os.path.isdir(video_folder):
            keyframes[video] = [os.path.join(video_folder, img) for img in os.listdir(video_folder)]
    return keyframes

def save_keyframe_data(keyframe_data, output_file):
    """Saves keyframe metadata to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(keyframe_data, f)

def load_keyframe_data(input_file):
    """Loads keyframe metadata from a JSON file."""
    with open(input_file, 'r') as f:
        return json.load(f)