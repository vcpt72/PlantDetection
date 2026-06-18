import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.ops import scale_boxes


YOLO_MODEL_PATH = 'runs/detect/yolov8-split/weights/best.pt' 
IMAGE_PATH = 'plates_imgs/img_023.png' #change for your needs
DISH_DIA_MM = 150.0  
OUTPUT_PATH = 'result_height_only.jpg'

print("Loading YOLO model...")
model = YOLO(YOLO_MODEL_PATH)

img = cv2.imread(IMAGE_PATH)
if img is None:
    print(f"Error: Image {IMAGE_PATH} not found.")
    exit()

h_orig, w_orig = img.shape[:2]


px_to_mm_ratio = DISH_DIA_MM / h_orig

img_viz = img.copy()


print("Detecting plants...")
results = model(img, conf=0.4, iou=0.6)
r = results[0]

if len(r.boxes) > 0:
   
    bboxes = r.boxes.xyxy.cpu().numpy()
    
    print(f"Found plants: {len(bboxes)}")

    for box in bboxes:
        x1, y1, x2, y2 = map(int, box)

        height_px = y2 - y1
        
        height_mm = height_px * px_to_mm_ratio

        cv2.rectangle(img_viz, (x1, y1), (x2, y2), (0, 0, 255), 2)
      
        label = f"{height_mm:.1f} mm"
        cv2.putText(img_viz, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    cv2.imwrite(OUTPUT_PATH, img_viz)
    print(f"Result in: {OUTPUT_PATH}")
else:
    print("YOLO didn't find anything.")
