import pandas as pd
import cv2
import os
import numpy as np
import glob

IMG_DIR = 'plates_imgs'          
CSV_DIR = 'plates_csv'          
OUT_DIR = 'yolo_dataset'   
TARGET_SIZE = 1024              
MARGIN = 30                     

os.makedirs(os.path.join(OUT_DIR, 'images'), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'csv'), exist_ok=True)

image_files = glob.glob(os.path.join(IMG_DIR, "*.png"))

print(f"Resizing {len(image_files)} images...")

for img_path in image_files:
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    csv_name = f"annotation-{base_name}.csv"
    csv_path = os.path.join(CSV_DIR, csv_name)
    
    if not os.path.exists(csv_path):
        continue

    img = cv2.imread(img_path)
    df = pd.read_csv(csv_path)
    if img is None or df.empty: continue

    h_orig, w_orig = img.shape[:2]

    x1_all, y1_all = df['bbox_x'].min(), df['bbox_y'].min() 
    x2_all = (df['bbox_x'] + df['bbox_width']).max()
    y2_all = (df['bbox_y'] + df['bbox_height']).max()

    x1_crop = max(0, int(x1_all - MARGIN))
    y1_crop = max(0, int(y1_all - MARGIN))
    x2_crop = min(w_orig, int(x2_all + MARGIN))
    y2_crop = min(h_orig, int(y2_all + MARGIN))

    cropped_img = img[y1_crop:y2_crop, x1_crop:x2_crop]
    ch, cw = cropped_img.shape[:2]

    scale = TARGET_SIZE / max(ch, cw)
    new_w, new_h = int(cw * scale), int(ch * scale)
    resized_img = cv2.resize(cropped_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    final_img = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
    dx = (TARGET_SIZE - new_w) // 2
    dy = (TARGET_SIZE - new_h) // 2
    final_img[dy:dy+new_h, dx:dx+new_w] = resized_img

    new_rows = []
    for _, row in df.iterrows():
        nx = int((row['bbox_x'] - x1_crop) * scale + dx)
        ny = int((row['bbox_y'] - y1_crop) * scale + dy)
        nw = int(row['bbox_width'] * scale)
        nh = int(row['bbox_height'] * scale)

        new_rows.append({
            'label_name': row['label_name'],
            'bbox_x': nx,
            'bbox_y': ny,
            'bbox_width': nw,
            'bbox_height': nh,
            'image_name': f"{base_name}.png", 
            'image_width': TARGET_SIZE,
            'image_height': TARGET_SIZE
        })

   
    cv2.imwrite(os.path.join(OUT_DIR, 'images', f"{base_name}.png"), final_img)

    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(os.path.join(OUT_DIR, 'csv', csv_name), index=False)

print(f"Resized dataset in '{OUT_DIR}'.")

