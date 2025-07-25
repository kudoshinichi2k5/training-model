import cv2
import os
import random
import albumentations as A

IMAGE_DIR = 'original_images/'
LABEL_DIR = 'original_labels/'
AUG_IMAGE_DIR = 'augmented_images/'
AUG_LABEL_DIR = 'augmented_labels/'
NUM_AUGMENTATIONS_PER_IMAGE = 30 # Số lượng ảnh tăng cường muốn tạo ra cho mỗi ảnh gốc

# Tạo thư mục đầu ra nếu chưa có
os.makedirs(AUG_IMAGE_DIR, exist_ok=True)
os.makedirs(AUG_LABEL_DIR, exist_ok=True)

# --- ĐỊNH NGHĨA CÁC PHÉP TĂNG CƯỜNG ---
# Tham khảo thêm tại: https://albumentations.ai/docs/
transform = A.Compose([
    A.Rotate(limit=15, p=0.7),      # Xoay ảnh ngẫu nhiên trong khoảng -15 đến 15 độ, xác suất 70%
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),    # Thay đổi độ sáng và độ tương phản ngẫu nhiên, xác suất 80%
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),   # Làm mờ ảnh bằng bộ lọc Gaussian, xác suất 50%
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),    # Thêm nhiễu Gaussian vào ảnh, xác suất 50%
    A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15, p=0.7, border_mode=cv2.BORDER_CONSTANT), # Dịch, thay đổi tỷ lệ và xoay ảnh, xác suất 70%
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Đọc file classes.txt để ánh xạ tên lớp sang ID
# Điều này giúp script hoạt động ngay cả khi bạn không biết thứ tự ID
try:
    with open(os.path.join(LABEL_DIR, 'classes.txt'), 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'classes.txt'. Hãy đảm bảo nó tồn tại trong thư mục nhãn.")
    exit()

class_map = {name: i for i, name in enumerate(class_names)}
print(f"Các lớp được tìm thấy: {class_map}")

# --- BẮT ĐẦU VÒNG LẶP TĂNG CƯỜNG ---
for image_filename in os.listdir(IMAGE_DIR):
    if not image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    image_path = os.path.join(IMAGE_DIR, image_filename)
    base_filename = os.path.splitext(image_filename)[0]
    label_filename = base_filename + '.txt'
    label_path = os.path.join(LABEL_DIR, label_filename)

    if not os.path.exists(label_path):
        print(f"Cảnh báo: Bỏ qua ảnh {image_filename} vì không có file nhãn tương ứng.")
        continue

    # Kiểm tra xem ảnh có chứa class 'cam_re_trai' không
    contains_cam_re_trai = any(int(line.strip().split()[0]) == 3 for line in open(label_path))
    if not contains_cam_re_trai:
        continue  # Bỏ qua nếu không chứa cam_re_trai


    # Đọc ảnh
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Đọc các bounding box từ file YOLO
    bboxes = []
    class_labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            bboxes.append([x_center, y_center, width, height])
            # Chuyển đổi ID về tên để albumentations hiểu
            class_labels.append(class_names[class_id])
    
    # Tạo các phiên bản tăng cường
    for i in range(NUM_AUGMENTATIONS_PER_IMAGE):
        augmented_data = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        
        aug_image = augmented_data['image']
        aug_bboxes = augmented_data['bboxes']
        
        # Tạo tên file mới
        new_filename_base = f"{base_filename}_aug_{i}"
        new_image_path = os.path.join(AUG_IMAGE_DIR, new_filename_base + '.jpg')
        new_label_path = os.path.join(AUG_LABEL_DIR, new_filename_base + '.txt')
        
        # Lưu ảnh mới
        cv2.imwrite(new_image_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
        
        # Lưu file nhãn mới
        with open(new_label_path, 'w') as f:
            for bbox, label in zip(aug_bboxes, augmented_data['class_labels']):
                # Chuyển tên lớp về lại ID
                class_id = class_map[label]
                x_center, y_center, width, height = bbox
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    print(f"Đã tạo {NUM_AUGMENTATIONS_PER_IMAGE} phiên bản cho ảnh {image_filename}")

print("\nHoàn tất quá trình tăng cường dữ liệu!")