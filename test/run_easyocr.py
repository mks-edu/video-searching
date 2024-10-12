import easyocr

reader = easyocr.Reader(['vi', 'en'])

image_frame = r"data\16.jpg"

img_text = reader.readtext(image_frame)
final_text = ""

for _, text, __ in img_text:  # _ = bounding box, text = text and __ = confident level
    final_text += text
    final_text += " "

print(final_text)
