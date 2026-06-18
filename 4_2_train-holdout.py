from ultralytics import YOLO


model = YOLO('yolov8m.pt')  

#model = YOLO('yolo11m.pt')  

#model = YOLO('yolo26m.pt')  

model.train(
    data='plants-split.yaml', 
    epochs=100,
    imgsz=1024,
    batch=2,
    device=0,
    name='yolov8-split', # name='yolov11-split',        # name='yolov26-split',
    patience=20
)