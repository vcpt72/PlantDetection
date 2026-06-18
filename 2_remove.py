import cv2
import numpy as np
import os
from pathlib import Path

INPUT_FOLDER = "yolo_dataset/images"
OUTPUT_FOLDER = "processed_plates_final"
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

def process_image(img_path, output_p):
    img = cv2.imread(str(img_path))
    if img is None: return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = gray.shape
    
    
    stred = gray[int(h*0.4):int(h*0.6), int(w*0.4):int(w*0.6)]

    dynamic_thresh = int(np.clip(np.mean(stred) - 25, 13, 65))
    #print(f"Zpracovano: {dynamic_thresh} ", end="\r")
    _, mask_gray = cv2.threshold(gray, dynamic_thresh, 255, cv2.THRESH_BINARY_INV)
    mask_purple = cv2.inRange(hsv, np.array([115, 30, 30]), np.array([175, 255, 200]))
    mask = cv2.bitwise_or(mask_gray, mask_purple)

  
    k_small = np.ones((3,3), np.uint8)
   
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))

   
   
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_small)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_h)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_v)
    
  
    mask = cv2.dilate(mask, k_small, iterations=2)


    result = cv2.inpaint(img, mask, 2, cv2.INPAINT_NS)

   
    out_p_png = output_p.with_suffix('.png')
    cv2.imwrite(str(out_p_png), result)



files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".png")]

print(f"Cisteni fixu")

for filename in sorted(files):
    process_image(Path(INPUT_FOLDER) / filename, Path(OUTPUT_FOLDER) / filename)
    print(f"Zpracovano: {filename} ", end="\r")

print(f"\n\nHotovo!")
