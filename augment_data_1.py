import cv2
import os
import random
import albumentations as A
from collections import defaultdict

# === CẤU HÌNH ===
IMAGE_DIR = 'original_images/'
LABEL_DIR = 'original_labels/'
AUG_IMAGE_DIR = 'augmented_images/'
AUG_LABEL_DIR = 'augmented_labels/'

# Số lượng augment mong muốn cho từng lớp (tính theo tổng số ảnh augment)
AUGMENTATION_MAP = {
    're_trai': 200,
    're_phai': 200,
    'cam_re_trai': 200,
    'cam_re_phai': 200,
    'di_thang': 100,
    'cam_di_thang': 100
}

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

# === TRANSFORMS ===
transform = A.Compose([
    A.Rotate(limit=10, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
    A.GaussianBlur(blur_limit=(3, 5), p=0.4),
    A.GaussNoise(var_limit=(10.0, 40.0), p=0.4),
    A.MotionBlur(blur_limit=3, p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=0, p=0.6, border_mode=cv2.BORDER_CONSTANT),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))

# === THU THẬP ẢNH CHỨA CLASS ===
class_to_images = defaultdict(list)

for image_filename in os.listdir(IMAGE_DIR):
    if not image_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    label_path = os.path.join(LABEL_DIR, image_filename.rsplit('.', 1)[0] + '.txt')
    if not os.path.exists(label_path):
        continue

    with open(label_path, 'r') as f:
        for line in f:
            cls_id = int(line.strip().split()[0])
            class_name = id_to_class.get(cls_id)
            if class_name in AUGMENTATION_MAP:
                class_to_images[class_name].append(image_filename)
                # không break, vì ảnh có thể chứa nhiều class

# === THỰC HIỆN AUGMENT CHO TỪNG CLASS ===
augment_counter = defaultdict(int)

for class_name, total_aug in AUGMENTATION_MAP.items():
    image_list = class_to_images.get(class_name, [])
    if not image_list:
        print(f"[⚠] Không tìm thấy ảnh nào chứa lớp '{class_name}' — bỏ qua.")
        continue

    print(f"[+] Augment lớp '{class_name}' với {total_aug} ảnh...")

    for i in range(total_aug):
        image_filename = random.choice(image_list)
        image_path = os.path.join(IMAGE_DIR, image_filename)
        label_path = os.path.join(LABEL_DIR, image_filename.rsplit('.', 1)[0] + '.txt')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = []
        class_labels = []
        with open(label_path, 'r') as f:
            for line in f:
                cls_id, x, y, bw, bh = line.strip().split()
                cls_id = int(cls_id)
                label_name = id_to_class[cls_id]
                if label_name == class_name:
                    bboxes.append([float(x), float(y), float(bw), float(bh)])
                    class_labels.append(label_name)

        if not bboxes:
            continue

        try:
            aug = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        except Exception as e:
            print(f"[!] Lỗi augment ảnh {image_filename}: {e}")
            continue

        aug_image = aug['image']
        aug_bboxes = aug['bboxes']
        aug_class_labels = aug['class_labels']

        # Gán tên file mới
        base_name = image_filename.rsplit('.', 1)[0]
        new_name = f"{class_name}_aug_{augment_counter[class_name]}"
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

        augment_counter[class_name] += 1

print("\n🎉 Đã hoàn tất tăng cường dữ liệu theo từng lớp:")
for cls, count in augment_counter.items():
    print(f"   • {cls}: {count} ảnh")
