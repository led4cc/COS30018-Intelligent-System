import os
import cv2
import torch
import numpy as np
from detect import detect
from models.experimental import attempt_load
from utils_LP import crop_n_rotate_LP
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

# Set environment variable to handle OpenMP runtime conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Configuration
Min_char = 0.01
Max_char = 0.09
image_path = 'data/test/images/test0.jpg'
LP_weights = 'LP_detect_yolov7_500img.pt'

# Load the character recognition model (PaddleOCR)
ocr = PaddleOCR(lang='en')

# Load the license plate detection model
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model_LP = attempt_load(LP_weights, map_location=device)

# Read the input image
source_img = cv2.imread(image_path)

# Detect license plates in the image
pred, LP_detected_img = detect(model_LP, source_img, device, imgsz=640)

print("Prediction structure:", pred)  # Debug statement

# Process each detected license plate
c = 0
for det in pred:
    for *xyxy, conf, cls in reversed(det):
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        angle, rotate_thresh, LP_rotated = crop_n_rotate_LP(source_img, x1, y1, x2, y2)
        if (rotate_thresh is None) or (LP_rotated is None):
            continue

        # Display the cropped and rotated license plate image
        cv2.imshow(f'Cropped and Rotated License Plate {c}', LP_rotated)

        # Preprocess the cropped and rotated license plate image
        gray = cv2.cvtColor(LP_rotated, cv2.COLOR_BGR2GRAY)
        cv2.imshow(f'Gray License Plate {c}', gray)

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow(f'Binary License Plate {c}', binary)

        # Convert numpy.ndarray to PIL Image for PaddleOCR
        LP_rotated_pil = Image.fromarray(binary)

        # Perform OCR using PaddleOCR
        result = ocr.ocr(np.array(LP_rotated_pil), cls=True)
        
        # Extract the recognized text
        recognized_text = ' '.join([line[1][0] for line in result[0]])
        print('Recognized License Plate:', recognized_text)

        # Annotate the image with the recognized text
        cv2.putText(LP_detected_img, recognized_text, (x1, y1 - 20), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)
        c += 1

print('Finally Done!')
cv2.imshow('Detected license plates', cv2.resize(LP_detected_img, dsize=None, fx=0.5, fy=0.5))
cv2.waitKey(0)
