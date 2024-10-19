import cv2
import pytesseract
import re
import unicodedata

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, lang='vie')

    # return preprocess_text(text)
    return text
def extract_text(image_path):
    img = cv2.imread(image_path)
    return extract_text_img(img)

# Process keyframes to extract text
def process_keyframes_for_text(keyframes):
    video_texts = {}
    for video, frames in keyframes.items():
        video_texts[video] = {}
        for frame in frames:
            text = extract_text(frame)
            video_texts[video][frame] = text
    return video_texts

def preprocess_text(text):
    """
    Preprocess the OCR-extracted text by cleaning and normalizing it.

    Args:
        text (str): The raw OCR-extracted text.

    Returns:
        str: The cleaned and normalized text.
    """
    # Normalize Unicode characters
    text = unicodedata.normalize('NFKD', text)

    # Remove non-printable characters
    text = ''.join(c for c in text if c.isprintable())

    # Remove unwanted symbols and punctuation
    # This regex removes any character that is not a letter, number, or space
    text = re.sub(r'[^A-Za-zÀ-ÿ0-9\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Optionally, convert text to lowercase for consistency
    text = text.lower()

    return text