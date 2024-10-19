import os
import chromadb
from chromadb_clip import VideoChromaDb
from keyframe.text import extract_text

def save_extract_text(extract_text_output_folder, video_cat, video_name, frame_name, text):
    # Write extracted text
    extract_text_video_output_folder = os.path.join(extract_text_output_folder, video_cat, video_name)
    if not os.path.exists(extract_text_video_output_folder):
        os.makedirs(extract_text_video_output_folder)

    extract_fpath = os.path.join(extract_text_video_output_folder, frame_name + '.txt')
    with open(extract_fpath, 'w') as file:
        # Write a string to the file
        file.write(text)

# Process keyframes and index data
def process_keyframes_for_index(keyframe_folder, object_folder, clip_feature_folder):
    vdb = VideoChromaDb('db_chromadb_video_all1')
    # video_cat: L29, L30
    for video_cat in os.listdir(keyframe_folder):
        cat_video_folder = os.path.join(keyframe_folder, video_cat)

        print(f'Processing folder of video category {cat_video_folder}')

        for video_name in os.listdir(cat_video_folder):

            video_folder = os.path.join(cat_video_folder, video_name)

            if os.path.isdir(video_folder):
                # frame_path[video_name] = [os.path.join(video_folder, filename) for filename in os.listdir(video_folder)]
                for filename in os.listdir(video_folder):
                    if not filename.lower().endswith('.jpg'):
                        continue

                    filename_no_ext = os.path.splitext(os.path.basename(filename))[0]

                    if vdb.hasFrame(video_cat, video_name, filename_no_ext):
                        continue

                    image_filepath = os.path.join(video_folder, filename)

                    print(f'Processing frame {image_filepath}')
                    video_feature_path = os.path.join(clip_feature_folder, video_cat, video_name + '.npy')
                    object_path = os.path.join(object_folder, video_cat, video_name, filename_no_ext + '.json')

                    text = extract_text(image_filepath)

                    save_extract_text(extract_text_output_folder, video_cat, video_name, filename_no_ext, text)

                    # video_cat, video_name, video_npy_file, image_filepath, json_file, extracted_text):
                    vdb.add_video_and_image_to_collection(video_cat, video_name, video_feature_path, image_filepath, object_path, text)

                    print('Done.')


# Example usage
# keyframe_folder = r"V:\AIC-2024\Data_2024\Keyframe\Keyframes"
object_folder = r'V:\AIC-2024\Data_2024\Objects\objects'
clip_feature_folder = r'V:\AIC-2024\Data_2024\CLIP-features\clip-features-32-b3\clip-features-32'

extract_text_output_folder = r'V:\AIC-2024\Data_2024\Keyframe_extracted_text'
# keyframe_folder = r'V:\AIC-2024\Data_2024\Keyframe\Keyframes_L28'
# keyframe_folder = r'V:\AIC-2024\Data_2024\Keyframe\Keyframes_L27'
# keyframe_folder = r'V:\AIC-2024\Data_2024\Keyframe\Keyframes_L26_e'
# keyframe_folder = r'V:\AIC-2024\Data_2024\Keyframe\Keyframes_L26_d'
# keyframe_folder = r'V:\AIC-2024\Data_2024\Keyframe\Keyframes_L26_c'
# keyframe_folder = r'V:\AIC-2024\Data_2024\Keyframe\Keyframes_L26_b'
# keyframe_folder = r'V:\AIC-2024\Data_2024\Keyframe\Keyframes_L26_a'
keyframe_folder = r'F:\KeyFrames_0.1'
process_keyframes_for_index(keyframe_folder, object_folder, clip_feature_folder)

# Save collection to disk (if persistence is supported by your ChromaDB version)
#save()
