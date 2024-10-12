import cv2
import pytesseract
# Initialize Tesseract OCR

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Extract text from images
def extract_text(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, lang='vie')
    return text

image_frame = r"data\16.jpg"
image_texts = extract_text(image_frame)

print(image_texts)
