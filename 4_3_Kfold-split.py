import os
import shutil
import glob
import numpy as np
from sklearn.model_selection import KFold


BASE_DATA = 'trainsplit/final_dataset'

DEST_ROOT = os.path.abspath('trainsplit/kfold_dataset')

def get_basenames(subset):
    path = os.path.join(BASE_DATA, subset, 'images', '*.png')
    return [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(path)]


train_names = get_basenames('train')
val_names = get_basenames('val')
all_train_val = np.array(train_names + val_names)

def copy_files(names, source_subset, target_path):
    os.makedirs(os.path.join(target_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(target_path, 'labels'), exist_ok=True)
    
    for name in names:
        #
        for s in ['train', 'val']:
            img_src = os.path.join(BASE_DATA, s, 'images', f"{name}.png")
            lbl_src = os.path.join(BASE_DATA, s, 'labels', f"{name}.txt")
            if os.path.exists(img_src):
                shutil.copy(img_src, os.path.join(target_path, 'images'))
                shutil.copy(lbl_src, os.path.join(target_path, 'labels'))
                break


kf = KFold(n_splits=3, shuffle=True, random_state=42)

for fold, (t_idx, v_idx) in enumerate(kf.split(all_train_val)):
    fold_dir = os.path.join(DEST_ROOT, f"fold_{fold}")
   
    f_train_names = all_train_val[t_idx]
    f_val_names = all_train_val[v_idx]

    copy_files(f_train_names, None, os.path.join(fold_dir, 'train'))
    copy_files(f_val_names, None, os.path.join(fold_dir, 'val'))
    
    
    yaml_content = {
        'path': fold_dir,
        'train': 'train/images',
        'val': 'val/images',
        'names': {0: 'seedling'}
    }
    
    import yaml
    with open(os.path.join(fold_dir, f"data_fold_{fold}.yaml"), 'w') as f:
        yaml.dump(yaml_content, f)

print(f"Hotovo! {DEST_ROOT}")