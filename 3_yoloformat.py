import pandas as pd
import os
import glob


INPUT_CSV_DIR = 'processed_dataset/csv'   
OUTPUT_TXT_DIR = 'yolo_labels'            
LABEL_MAP = {'seedling': 0}               

os.makedirs(OUTPUT_TXT_DIR, exist_ok=True)

csv_files = glob.glob(os.path.join(INPUT_CSV_DIR, "*.csv"))

print(f"Converting {len(csv_files)} files...")

for csv_path in csv_files:
   
    df = pd.read_csv(csv_path)
    if df.empty:
        continue
    
    
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
   
    txt_name = base_name.replace('annotation-', '') + ".txt"
    
    yolo_lines = []
    
    for _, row in df.iterrows():
        
        img_w = row['image_width']
        img_h = row['image_height']
        
       
        x_center = (row['bbox_x'] + row['bbox_width'] / 2.0) / img_w
        y_center = (row['bbox_y'] + row['bbox_height'] / 2.0) / img_h
        
       
        w_norm = row['bbox_width'] / img_w
        h_norm = row['bbox_height'] / img_h
        
       
        class_id = LABEL_MAP.get(row['label_name'], 0)
        
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
    
    with open(os.path.join(OUTPUT_TXT_DIR, txt_name), 'w') as f:
        f.write("\n".join(yolo_lines))

print(f"Done, files in '{OUTPUT_TXT_DIR}'.")
