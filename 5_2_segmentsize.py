#Vypocet delky rostlin promoci skeletonizace a bfs
#Size of plants measured by skeletonization and bfs
import cv2
import numpy as np
from ultralytics import YOLO, SAM
from ultralytics.utils.ops import scale_boxes
from skimage.morphology import skeletonize
from collections import deque
import math

# Config
device = 'cpu' # or cuda
YOLO_MODEL_PATH = 'runs/detect/yolov11-split/weights/best.pt'
SAM_MODEL_PATH = 'sam2_t.pt'
IMAGE_PATH = 'plates_imgs/img_001.png'
DISH_DIA_MM = 150.0 
OUTPUT_PATH = 'measurement.jpg'

def find_path_bfs(skeleton, start_node, end_node):
    
    pixel_set = set(map(tuple, np.column_stack(np.where(skeleton > 0))))
    if not pixel_set: return None

    q = deque([(start_node, [start_node])])
    visited = {start_node}

    # (vodorovne, svisle, diagonalne)
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),               # (-1, -1), (-1, 0), (-1, 1), 
        (0, -1),           (0, 1),                # (0, -1),   PIXEL   (0, 1), 
        (1, -1),  (1, 0),  (1, 1)                 # (1, -1),  (1, 0),  (1, 1)
    ]
    
    while q:
        (y, x), path = q.popleft()

        if (y, x) == end_node:
            return path

        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if (ny, nx) in pixel_set and (ny, nx) not in visited:
                visited.add((ny, nx))
                q.append(((ny, nx), path + [(ny, nx)]))
    
    return None

print("Loading models...")
detector = YOLO(YOLO_MODEL_PATH)
segmenter = SAM(SAM_MODEL_PATH).to(device)

img = cv2.imread(IMAGE_PATH)
if img is None: exit()

h_orig, w_orig = img.shape[:2]
px_to_mm_ratio = DISH_DIA_MM / w_orig
img_viz = img.copy()

results_yolo = detector(img, conf=0.4, iou=0.6)

if len(results_yolo[0].boxes) > 0:
    bboxes_yolo = scale_boxes(results_yolo[0].orig_shape, results_yolo[0].boxes.xyxy.clone(), img.shape).cpu().numpy()
    results_sam = segmenter.predict(img, bboxes=bboxes_yolo.astype(np.float32), device=device, verbose=False)
    masks_data = results_sam[0].masks.data.numpy()

    kernel = np.ones((15, 15), np.uint8)

    for i in range(len(bboxes_yolo)):
        x1, y1, x2, y2 = map(int, bboxes_yolo[i])
        
        # 1. Morfologie
        mask_raw = (masks_data[i] > 0.3).astype(np.uint8)
        if mask_raw.shape != (h_orig, w_orig):
            mask_raw = cv2.resize(mask_raw, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        mask_dilated = cv2.dilate(mask_raw, kernel, iterations=3)
        mask_closed = cv2.morphologyEx(mask_dilated, cv2.MORPH_CLOSE, kernel)
        mask_connected = cv2.erode(mask_closed, kernel, iterations=2)

        # 2. Skeletonizace
        skel = skeletonize(mask_connected > 0).astype(np.uint8)
        points = np.column_stack(np.where(skel > 0))
        
        length_mm = 0.0
        path = None
        color_label = (0, 255, 0)
        is_estimate = False

        if len(points) > 5:
            start = tuple(points[np.argmin(points[:, 0])]) # min y 
            end = tuple(points[np.argmax(points[:, 0])])   # max y
            path = find_path_bfs(skel, start, end)

        # 3. Sizing
        if path:
            path_pts = np.array(path)
            pixel_length = np.sum(np.sqrt(np.sum(np.diff(path_pts, axis=0)**2, axis=1)))
            length_mm = pixel_length * px_to_mm_ratio
            
            # Vykresleni
            for j in range(len(path_pts) - 1):
                cv2.line(img_viz, tuple(path_pts[j][::-1]), tuple(path_pts[j+1][::-1]), (255, 0, 255), 2)
        else:
            # Fallback bounding box height
            pixel_length = abs(y2 - y1)
            length_mm = pixel_length * px_to_mm_ratio
            color_label = (0, 165, 255) 
            is_estimate = True

        # 4. Vizualizace
        cv2.rectangle(img_viz, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        label_text = f"{length_mm:.1f} mm"
        

        cv2.putText(img_viz, label_text, (x1, max(y1 - 10, 30)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_label, 2)

    cv2.imwrite(OUTPUT_PATH, img_viz)
    print(f"Detected and measured plants in {OUTPUT_PATH}")
