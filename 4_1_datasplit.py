import os
import shutil
import glob
from sklearn.model_selection import train_test_split


IMG_SOURCE = 'trainsplit/processed_plates_final'
LABEL_SOURCE = 'trainsplit/labels'
DEST_ROOT = 'trainsplit/final_dataset'


images = glob.glob(os.path.join(IMG_SOURCE, "*.png"))

img_basenames = [os.path.splitext(os.path.basename(f))[0] for f in images]

# 15% test
train_val_names, test_names = train_test_split(img_basenames, test_size=0.15, random_state=42)

# 15% validation # rest train
train_names, val_names = train_test_split(train_val_names, test_size=0.176, random_state=42)

def copy_files(names, split):
    for name in names:
        
        src_img = os.path.join(IMG_SOURCE, f"{name}.png")
        src_lbl = os.path.join(LABEL_SOURCE, f"{name}.txt")
        
        if os.path.exists(src_lbl):
            
            dst_img_dir = os.path.join(DEST_ROOT, split, 'images')
            dst_lbl_dir = os.path.join(DEST_ROOT, split, 'labels')
            os.makedirs(dst_img_dir, exist_ok=True)
            os.makedirs(dst_lbl_dir, exist_ok=True)
            
            shutil.copy(src_img, dst_img_dir)
            shutil.copy(src_lbl, dst_lbl_dir)


copy_files(train_names, 'train')
copy_files(val_names, 'val')
copy_files(test_names, 'test')
print(f"Splitted: {len(train_names)} / {len(val_names)} / {len(test_names)}")
