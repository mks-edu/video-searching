import os

from keyframe.frame import load_keyframe_data
from keyframe.frame import save_keyframe_data

def process_keyfram_data_step1(folder_path):
    '''
    Combine all data of keyframes into one file
    :param folder_path:
    :return: combined data
    '''
    combined_data = {}
    for path, folders, files in os.walk(folder_path):
        for f in files:
            file_path = os.path.join(path, f)
            print(f'Processing file {file_path}...')

            frame_data_json = load_keyframe_data(file_path)
            # Loop through the dictionary
            for key, value in frame_data_json.items():
                combined_data[key] = value

    return combined_data

# Data of each keyframe is processing in step 1.
keyframe_data_folder = r"G:\Projects\github.com\mks-edu\video-searching\keyframe_data_step1"

all_keyframe_data = process_keyfram_data_step1(keyframe_data_folder)

# Save file all combined data
save_keyframe_data(all_keyframe_data, 'all_keyframe_data.json')

