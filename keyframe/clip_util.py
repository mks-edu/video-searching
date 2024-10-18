import torch
import clip
from PIL import Image
from transformers import CLIPTokenizer

# Load CLIP model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def extract_image_embedding(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = model.encode_image(image).cpu().numpy()
    return image_embedding

def extract_text_embedding(text):
    text_tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens).cpu().numpy()
    return text_embedding

def truncate_text_to_token_limit(text, clip_model, max_token_length=77):
    """
    Ensures the text is within CLIP's maximum token limit by truncating if necessary.
    """
    tokens = clip.tokenize([text])[0]  # Tokenize once to get the token length
    if len(tokens) > max_token_length:
        # If the number of tokens exceeds the limit, truncate the text
        words = text.split()
        truncated_text = ' '.join(words[:max_token_length])
        return truncated_text
    return text

def split_text_into_token_chunks(text, max_token_length=77):
    tokens = CLIPTokenizer.encode(text)
    # Subtract 2 to account for special tokens added by the tokenizer
    max_tokens = max_token_length - 2
    token_chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    text_chunks = [CLIPTokenizer.decode(chunk) for chunk in token_chunks]
    return text_chunks