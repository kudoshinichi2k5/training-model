import cv2
import os
import random
import albumentations as A

# === CẤU HÌNH ===
IMAGE_DIR = 'original_images/'
LABEL_DIR = 'original_labels/'
AUG_IMAGE_DIR = 'augmented_images/'
AUG_LABEL_DIR = 'augmented_labels/'
NUM_AUGMENTATIONS_PER_IMAGE = 30

# Các class cần tăng cường
TARGET_CLASSES = ['re_trai', 're_phai', 'cam_re_trai', 'cam_re_phai']

# Tạo thư mục output
os.makedirs(AUG_IMAGE_DIR, exist_ok=True)
os.makedirs(AUG_LABEL_DIR, exist_ok=True)

# === LOAD CLASS NAMES ===
try:
    with open(os.path.join(LABEL_DIR, 'classes.txt'), 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'classes.txt' trong thư mục nhãn.")
    exit()

class_map = {name: i for i, name in enumerate(class_names)}
id_to_class = {i: name for name, i in class_map.items()}

print(f"Đã tìm thấy các lớp: {class_map}")
target_class_ids = [class_map[name] for name in TARGET_CLASSES if name in class_map]

# === TRANSFORMS ===
transform = A.Compose([
    A.Rotate(limit=10, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
    A.GaussianBlur(blur_limit=(3, 5), p=0.4),
    A.GaussNoise(var_limit=(10.0, 40.0), p=0.4),
    A.MotionBlur(blur_limit=3, p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=0, p=0.6, border_mode=cv2.BORDER_CONSTANT),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))

# === AUGMENTATION LOOP ===
for image_filename in os.listdir(IMAGE_DIR):
    if not image_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(IMAGE_DIR, image_filename)
    label_path = os.path.join(LABEL_DIR, image_filename.rsplit('.', 1)[0] + '.txt')

    if not os.path.exists(label_path):
        print(f"Bỏ qua {image_filename} vì không có file nhãn.")
        continue

    # Load ảnh và nhãn
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    bboxes = []
    class_labels = []
    with open(label_path, 'r') as f:
        for line in f:
            cls_id, x, y, bw, bh = line.strip().split()
            cls_id = int(cls_id)
            if cls_id in target_class_ids:
                bboxes.append([float(x), float(y), float(bw), float(bh)])
                class_labels.append(id_to_class[cls_id])

    if not bboxes:
        continue  # Không augment nếu ảnh không chứa class mục tiêu

    for i in range(NUM_AUGMENTATIONS_PER_IMAGE):
        try:
            aug = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        except Exception as e:
            print(f"Lỗi khi augment {image_filename}: {e}")
            continue

        aug_image = aug['image']
        aug_bboxes = aug['bboxes']
        aug_class_labels = aug['class_labels']

        # Gán tên file mới
        new_name = image_filename.rsplit('.', 1)[0] + f'_aug_{i}'
        new_image_path = os.path.join(AUG_IMAGE_DIR, new_name + '.jpg')
        new_label_path = os.path.join(AUG_LABEL_DIR, new_name + '.txt')

        # Lưu ảnh
        cv2.imwrite(new_image_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

        # Lưu nhãn
        with open(new_label_path, 'w') as f:
            for bbox, label in zip(aug_bboxes, aug_class_labels):
                class_id = class_map[label]
                x, y, bw, bh = bbox
                if 0 <= x <= 1 and 0 <= y <= 1 and 0 < bw <= 1 and 0 < bh <= 1:
                    f.write(f"{class_id} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"✓ {image_filename} → {NUM_AUGMENTATIONS_PER_IMAGE} bản tăng cường.")

print("\n🎉 Hoàn tất quá trình tăng cường cho các lớp dễ nhầm.")
