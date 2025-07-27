yolo detect train data=traffic_signs.yaml model=yolov8n.pt epochs=100 imgsz=640
!yolo detect train data=traffic_signs.yaml model=yolov8s.pt epochs=150 imgsz=640 batch=32 conf=0.4 iou=0.5 device=0