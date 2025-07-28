from ultralytics import YOLO
import os

# Đường dẫn model và thư mục ảnh test
model_path = 'runs/detect/train/weights/best.pt'
test_images_dir = 'dataset/images/val'  # Thay bằng đường dẫn thư mục ảnh test của bạn

# Load model
model = YOLO(model_path)

# Lấy danh sách file ảnh
image_files = [os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Dự đoán hàng loạt và lưu kết quả vào runs/detect/predict/
results = model.predict(source=image_files, save=True, project='runs/detect', name='predict', exist_ok=True)

print("Đã lưu kết quả vào runs/detect/predict/")