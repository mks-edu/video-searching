import os
import chromadb
import numpy as np
import torch
import clip
import json
from PIL import Image
from keyframe.transformer_util import MAX_TOKEN_LENGTH, extract_text_embedding, extract_long_text_embedding

collection_names = ['db_video_embedding', 'db_keyframe_embedding', 'db_video_text_embedding', 'db_keyframe_text_embedding']
class VideoChromaDb():

    def __init__(self, db_path):
        '''
        db_chromadb_video_search_extracted_text
        datatype: 0, 1, 2
        '''

        # Load the CLIP model (using the 'ViT-B/32' variant)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        # Initialize ChromaDB client
        # client = chromadb.Client()
        client = chromadb.PersistentClient(path=db_path)

        # Try to Reload collection
        # collection_names = ['db_video_embedding', 'db_keyframe_embedding', 'db_video_text_embedding', 'db_keyframe_text_embedding']

        self.collections = {}

        for collection_name in collection_names:
            # Check if collection already exists
            try:
                self.collections[collection_name] = client.get_collection(collection_name)
                print(f"Collection '{collection_name}' loaded successfully.")
            except:
                self.collections[collection_name] = client.create_collection(collection_name)
                print(f"Collection '{collection_name}' created.")

    def hasFrame(self, video_cat, video_name, frame_name):
        image_id = f"{video_cat}_{video_name}_{frame_name}"

        for collection_name, collection in self.collections.items():
            result = collection.get(ids=[image_id])
            if result is not None and result['embeddings'] is not None:
                return True

        return False

    # Function to add data to the collection
    def add_video_and_image_to_collection(self, video_cat, video_name, video_npy_file, image_filepath, json_file, extracted_text):
        '''

        :param video_cat:
        :param video_name:
        :param frame_name: image file without extension
        :param npy_file:
        :param json_file:
        :param txt_file:
        :return:
        '''
        frame_name = os.path.splitext(os.path.basename(image_filepath))[0]
        # Load and preprocess the image file before embedding into CLIP
        image = Image.open(image_filepath)
        image_preprocessed = self.preprocess(image).unsqueeze(0).to(self.device)

        # Generate image feature features from the image
        with torch.no_grad():
            features_image = self.model.encode_image(image_preprocessed).cpu().numpy()

        # Load CLIP features for the video from .npy file (video-level embedding)
        if os.path.exists(video_npy_file):
            video_features = np.load(video_npy_file)

        # Load detection data from .json file
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                detection_data = json.load(f)

        # Embed the extracted OCR text using CLIP
        if len(extracted_text) < MAX_TOKEN_LENGTH:
            # text_tokens = clip.tokenize([extracted_text]).to(device)
            # with torch.no_grad():
            #     image_features_text = model.encode_text(text_tokens).cpu().numpy()
            extracted_text_image_features = extract_text_embedding(extracted_text)
        else:
            extracted_text_image_features = extract_long_text_embedding(extracted_text)


        # Convert lists in detection_data to JSON strings
        metadata = {
            # 'detection_scores': json.dumps(detection_data['detection_scores']),  # Convert list to string
            # 'detection_class_names': json.dumps(detection_data['detection_class_names']),  # Convert list to string
            # 'detection_class_entities': json.dumps(detection_data['detection_class_entities']),  # Convert list to string
            # 'detection_boxes': json.dumps(detection_data['detection_boxes']),  # Convert list to string
            # 'detection_class_labels': json.dumps(detection_data['detection_class_labels']),  # Convert list to string
            'ocr_text': extracted_text,  # Already a string
            'video_cat': video_cat,  # Already a string
            'video_name': video_name,  # Already a string
            'frame_name': frame_name  # Already a string
        }


        # Option 1: Store combined embeddings (image + video + text)
        #combined_embedding = np.concatenate((features_image, video_features_video, image_features_text), axis=1)
        # combined_embedding = np.concatenate((features_image, image_features_text), axis=1)

        # Add data to the collection
        # Construct a unique ID for the image
        image_id = f"{video_cat}_{video_name}_{frame_name}"
        video_id = f"{video_cat}_{video_name}"



        # Add collection of video_embedding
        # self.collections[collection_names[0]].add(
        #     ids=[video_id],
        #     embeddings=video_features.tolist(),
        #     metadatas=[]
        # )

        # Add collection of image_embedding
        self.collections[collection_names[1]].add(
            ids=[image_id],  # Unique ID for the image
            documents=[extracted_text],
            embeddings=features_image.tolist(),  # Add CLIP embeddings
            metadatas=[metadata]  # Add metadata including detection and OCR results
        )

        # Add collection of video text _embedding
        # self.collections[collection_names[2]].add(
        #     ids=[image_id],  # Unique ID for the image
        #     embeddings=features_image.tolist(),  # Add CLIP embeddings
        #     metadatas=[metadata]  # Add metadata including detection and OCR results
        # )

        # Add collection of image text_embedding
        self.collections[collection_names[3]].add(
            ids=[image_id],  # Unique ID for the image
            documents=[extracted_text],
            embeddings=extracted_text_image_features.tolist(),  # Add CLIP embeddings
            metadatas=[metadata]  # Add metadata including detection and OCR results
        )

        print(f"Added image '{image_id}' to collection.")

    def search_by_text(self, query_text, collection_id=3, n_results=5):
        '''

        :param query_text:
        :param collection_id: 3 search from extracted text of keyframe.
        :param n_results:
        :return:
        '''
        # Tokenize and generate CLIP embedding for the query text
        text_tokens = clip.tokenize([query_text]).to(self.device)
        with torch.no_grad():
            query_embedding = self.model.encode_text(text_tokens).cpu().numpy()

        # Query the collection for similar images/text
        results = self.collections[collection_names[collection_id]].query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )

        # Display results
        # for result in results['ids']:
        #     print(f"Found match: {result}")

        return results