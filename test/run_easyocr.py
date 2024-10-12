import easyocr

reader = easyocr.Reader(['vi', 'en'])

img_path = image_frame = r"\Keyframe\Keyframes_L01\keyframes\L01_V001\032.jpg"

img_text = reader.readtext(img_path)
final_text = ""

for _, text, __ in img_text:  # _ = bounding box, text = text and __ = confident level
    final_text += text
    final_text += " "

print(final_text)
