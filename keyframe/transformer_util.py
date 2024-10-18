from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import torch
from PIL import Image
import cv2

# Load CLIP model and tokenizer from transformers
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

MAX_TOKEN_LENGTH = 77  # Default token limit for CLIP

# Function to extract image embedding using CLIP
def extract_image_embedding(image_path):
    image = Image.open(image_path)
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_embedding = clip_model.get_image_features(**inputs).cpu().numpy()
    return image_embedding

# Function to handle long text inputs for text embeddings
def extract_text_embedding(text, max_length=MAX_TOKEN_LENGTH):
    inputs = clip_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        text_embedding = clip_model.get_text_features(**inputs).cpu().numpy()
    return text_embedding

# Optionally handle chunking of long text if truncation isn't desired
def extract_long_text_embedding(text, chunk_size=MAX_TOKEN_LENGTH):
    words = text.split()
    num_chunks = len(words) // chunk_size + (len(words) % chunk_size > 0)

    embeddings = []
    for i in range(num_chunks):
        chunk = " ".join(words[i*chunk_size : (i+1)*chunk_size])
        embedding = extract_text_embedding(chunk)
        embeddings.append(embedding)

    # Combine embeddings (e.g., average) for the final result
    combined_embedding = sum(embeddings) / len(embeddings)
    return combined_embedding
