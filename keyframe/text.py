import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, lang='vie')
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
