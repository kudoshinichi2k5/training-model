import paho.mqtt.client as mqtt
import cv2
import base64
import json
import time
import argparse
import uuid
import numpy as np
from collections import deque

MQTT_BROKER = 'nozomi.proxy.rlwy.net'  # Địa chỉ MQTT Broker
MQTT_PORT = 32067
REQUEST_TOPIC = "yolo/detect/request"
RESPONSE_TOPIC_PREFIX = "yolo/detect/response/"
CLIENT_ID = str(uuid.uuid4())
RESPONSE_TOPIC = RESPONSE_TOPIC_PREFIX + CLIENT_ID
# -----------------

is_connected = False
latest_response = None
last_print_time = 0

def on_connect(client, userdata, flags, rc, properties=None):
    global is_connected
    if rc == 0:
        print(f"Client ({CLIENT_ID}) đã kết nối!")
        client.subscribe(RESPONSE_TOPIC)
        is_connected = True
    else:
        print(f"Kết nối thất bại, mã lỗi: {rc}")
        is_connected = False

def on_message(client, userdata, msg):
    global latest_response
    latest_response = json.loads(msg.payload.decode())

def send_request(client, image_path_or_frame):
    if isinstance(image_path_or_frame, str): # Nếu là đường dẫn file (ảnh)
        with open(image_path_or_frame, "rb") as f:
            image_bytes = f.read()
    else: # Nếu là khung hình video (mảng numpy)
        _, buffer = cv2.imencode('.jpg', image_path_or_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_bytes = buffer.tobytes()

    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    payload = json.dumps({"client_id": CLIENT_ID, "image_b64": image_b64})
    client.publish(REQUEST_TOPIC, payload)

def draw_and_print_detections(image, detections):
    """Hàm chung để vẽ và in kết quả."""
    global last_print_time
    # Chỉ in ra terminal mỗi 1 giây để tránh làm ngập log
    current_time = time.time()
    if (current_time - last_print_time) > 1:
        print("-" * 30)
        if not detections:
            print("Không phát hiện đối tượng nào trong khung hình này.")
        else:
            print(f"Tìm thấy {len(detections)} đối tượng:")
            for i, det in enumerate(detections):
                class_name = det['class_name']
                confidence = det['confidence']
                print(f"  {i+1}. Nhãn: {class_name}, Độ tin cậy: {confidence * 100:.2f}%")
        last_print_time = current_time

    if detections:
        for det in detections:
            box = det['box']
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Client nhận diện đối tượng qua MQTT.")
    parser.add_argument('--mode', type=str, required=True, choices=['image', 'video'], help="Chế độ hoạt động: 'image' hoặc 'video'.")
    parser.add_argument('--source', type=str, required=True, help="Đường dẫn đến file ảnh/video, hoặc '0' cho webcam.")
    args = parser.parse_args()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)

    try:
        client.loop_start()

        print("Đang chờ kết nối đến MQTT Broker...")
        connection_start_time = time.time()
        while not is_connected and (time.time() - connection_start_time) < 5:
            time.sleep(0.1)

        if not is_connected:
            print("Không thể kết nối đến MQTT Broker. Đang thoát.")
            exit()
            
        if args.mode == 'image':
            image = cv2.imread(args.source)
            if image is None:
                print(f"Lỗi: Không thể đọc file ảnh '{args.source}'")
                exit()
            
            print(f"Kết nối thành công. Gửi yêu cầu xử lý ảnh: {args.source}")
            send_request(client, args.source)
            
            response_start_time = time.time()
            while latest_response is None and (time.time() - response_start_time) < 10:
                time.sleep(0.1)
                
            if latest_response:
                annotated_image = draw_and_print_detections(image, latest_response)
                cv2.imshow("Ket qua", annotated_image)
                cv2.waitKey(0)
            else:
                print("Không nhận được phản hồi từ server.")

        elif args.mode == 'video':
            source = 0 if args.source == '0' else args.source
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print(f"Lỗi: không thể mở nguồn video '{args.source}'")
                exit()
            
            print("Kết nối thành công. Bắt đầu stream video...")
            frame_interval = 1 / 20
            last_sent_time = 0
            # Giữ lại một vài kết quả cuối cùng để vẽ nếu chưa có kết quả mới
            last_known_detections = deque(maxlen=1) 

            while cap.isOpened():
                success, frame = cap.read()
                if not success: break

                current_time = time.time()
                if (current_time - last_sent_time) > frame_interval:
                    send_request(client, frame)
                    last_sent_time = current_time

                # Cập nhật kết quả cuối cùng nếu có phản hồi mới
                if latest_response is not None:
                    last_known_detections.append(latest_response)
                    latest_response = None # Reset để chờ phản hồi mới

                # Vẽ kết quả cuối cùng lên khung hình hiện tại
                if last_known_detections:
                    frame = draw_and_print_detections(frame, last_known_detections[0])

                cv2.imshow("Ket qua Video", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()

    except KeyboardInterrupt:
        print("\nĐã nhận tín hiệu dừng (Ctrl+C).")
    finally:
        print("Stop...")
        client.loop_stop()
        cv2.destroyAllWindows()
        print("Chương trình kết thúc.")