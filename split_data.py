import os
import random
import shutil

# Đường dẫn đến thư mục chứa tất cả ảnh và nhãn
SOURCE_IMAGES_DIR = "all_images/"
SOURCE_LABELS_DIR = "all_labels/"

# Đường dẫn đến thư mục đích (dataset)
DEST_DIR = "dataset/"

# Tỷ lệ phần trăm dữ liệu cho tập validation (ví dụ: 0.2 là 20%)
VAL_SPLIT_RATIO = 0.2


def split_data():
    """
    Hàm chính để chia dữ liệu thành tập train và validation.
    """
    print("Bắt đầu quá trình chia dữ liệu...")

    # 1. Tạo các thư mục đích nếu chúng chưa tồn tại
    train_img_path = os.path.join(DEST_DIR, 'images', 'train')
    val_img_path = os.path.join(DEST_DIR, 'images', 'val')
    train_label_path = os.path.join(DEST_DIR, 'labels', 'train')
    val_label_path = os.path.join(DEST_DIR, 'labels', 'val')

    os.makedirs(train_img_path, exist_ok=True)
    os.makedirs(val_img_path, exist_ok=True)
    os.makedirs(train_label_path, exist_ok=True)
    os.makedirs(val_label_path, exist_ok=True)
    print("Đã tạo cấu trúc thư mục đích.")

    # 2. Lấy danh sách tất cả các file ảnh
    all_images = [f for f in os.listdir(SOURCE_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not all_images:
        print(f"Lỗi: Không tìm thấy ảnh nào trong thư mục '{SOURCE_IMAGES_DIR}'.")
        return
        
    print(f"Tìm thấy tổng cộng {len(all_images)} ảnh.")

    # 3. Xáo trộn danh sách ảnh để đảm bảo tính ngẫu nhiên
    random.seed(42)  # Đặt seed để kết quả có thể lặp lại
    random.shuffle(all_images)
    print("Đã xáo trộn danh sách ảnh.")

    # 4. Tính toán điểm chia
    split_point = int(len(all_images) * (1 - VAL_SPLIT_RATIO))
    train_files = all_images[:split_point]
    val_files = all_images[split_point:]

    print(f"Chia dữ liệu: {len(train_files)} file cho tập train, {len(val_files)} file cho tập validation.")

    # 5. Hàm trợ giúp để di chuyển file
    def move_files(file_list, dest_img_path, dest_label_path):
        moved_count = 0
        for filename in file_list:
            base_filename = os.path.splitext(filename)[0]
            label_filename = base_filename + '.txt'
            
            source_img = os.path.join(SOURCE_IMAGES_DIR, filename)
            source_label = os.path.join(SOURCE_LABELS_DIR, label_filename)
            
            dest_img = os.path.join(dest_img_path, filename)
            dest_label = os.path.join(dest_label_path, label_filename)
            
            # Kiểm tra xem cả file ảnh và nhãn đều tồn tại trước khi di chuyển
            if os.path.exists(source_img) and os.path.exists(source_label):
                shutil.move(source_img, dest_img)
                shutil.move(source_label, dest_label)
                moved_count += 1
            else:
                print(f"Cảnh báo: Bỏ qua '{filename}' vì không tìm thấy cặp file ảnh/nhãn tương ứng.")
        return moved_count

    # 6. Di chuyển file vào thư mục train và val
    print("\nBắt đầu di chuyển file vào tập train...")
    train_moved = move_files(train_files, train_img_path, train_label_path)
    print(f"Đã di chuyển {train_moved} cặp file vào tập train.")
    
    print("\nBắt đầu di chuyển file vào tập validation...")
    val_moved = move_files(val_files, val_img_path, val_label_path)
    print(f"Đã di chuyển {val_moved} cặp file vào tập validation.")

    print("\nHoàn tất! Dữ liệu đã được chia thành công.")
    print(f"Kiểm tra thư mục '{DEST_DIR}' để xem kết quả.")


if __name__ == "__main__":
    split_data()