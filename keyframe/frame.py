import os

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