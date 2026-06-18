from ultralytics import YOLO


#model = YOLO('runs/detect/yolov8-split/weights/best.pt')
#model = YOLO('runs/detect/yolov11-split/weights/best.pt')
#model = YOLO('runs/detect/yolov26-split/weights/best.pt')
#model = YOLO('runs/detect/kfoldsplitv11/yolo11m_fold_0/weights/best.pt')
#model = YOLO('runs/detect/kfoldsplitv11/yolo11m_fold_1/weights/best.pt')
model = YOLO('runs/detect/kfoldsplitv11/yolo11m_fold_2/weights/best.pt')

metrics = model.val(data='plants-split.yaml', split='test', batch=1, conf=0.4)


print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
