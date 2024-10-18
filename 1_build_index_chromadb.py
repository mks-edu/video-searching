import os
import chromadb
from chromadb_clip import VideoChromaDb
from keyframe.text import extract_text

# Process keyframes and index data
def process_keyframes_for_index(keyframe_folder, object_folder, clip_feature_folder):
    vdb = VideoChromaDb('db_chromadb_video4')
    # video_cat: L29, L30
    for video_cat in os.listdir(keyframe_folder):
        cat_video_folder = os.path.join(keyframe_folder, video_cat)

        print(f'Processing folder of video category ${cat_video_folder}')

        for video_name in os.listdir(cat_video_folder):

            video_folder = os.path.join(cat_video_folder, video_name)

            if os.path.isdir(video_folder):
                # frame_path[video_name] = [os.path.join(video_folder, filename) for filename in os.listdir(video_folder)]
                for filename in os.listdir(video_folder):
                    filename_no_ext = os.path.splitext(os.path.basename(filename))[0]
                    image_filepath = os.path.join(video_folder, filename)

                    print(f'Processing frame {image_filepath}')
                    video_feature_path = os.path.join(clip_feature_folder, video_cat, video_name + '.npy')
                    object_path = os.path.join(object_folder, video_cat, video_name, filename_no_ext + '.json')

                    text = extract_text(image_filepath)
                    # video_cat, video_name, video_npy_file, image_filepath, json_file, extracted_text):
                    vdb.add_video_and_image_to_collection(video_cat, video_name, video_feature_path, image_filepath, object_path, text)

                    print('Done.')


# Example usage
# keyframe_folder = r"V:\AIC-2024\Data_2024\Keyframe\Keyframes"
object_folder = r'V:\AIC-2024\Data_2024\Objects\objects'
clip_feature_folder = r'V:\AIC-2024\Data_2024\CLIP-features\clip-features-32-b3\clip-features-32'

# keyframe_folder = r'V:\AIC-2024\Data_2024\Keyframe\Keyframes_L28'
keyframe_folder = r'V:\AIC-2024\Data_2024\Keyframe\Keyframes_L27'
process_keyframes_for_index(keyframe_folder, object_folder, clip_feature_folder)

# Save collection to disk (if persistence is supported by your ChromaDB version)
#save()
