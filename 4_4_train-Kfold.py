from ultralytics import YOLO
import os


KFOLD_ROOT = 'trainsplit/kfold_dataset'

FOLDS = [0, 1, 2] 



def train_kfold():
    for fold in FOLDS:
        print(f"\n" + "="*50)
        print(f"Training begins: FOLD {fold}")
        print("="*50 + "\n")
        
        yaml_path = os.path.join(KFOLD_ROOT, f"fold_{fold}", f"data_fold_{fold}.yaml")
        
       
        model = YOLO("yolo11m.pt")
        
    
        results = model.train(
            data=yaml_path,
            epochs=100,      
            imgsz=1024,       
            batch=2,       
            project="kfoldsplitv11",
            name=f"yolo11m_fold_{fold}",
            device=0,        
            patience=20      
        )
        
        print(f"Fold {fold} ended. Results in {PROJECT_NAME}/yolo11m_fold_{fold}")

if __name__ == "__main__":
    train_kfold()