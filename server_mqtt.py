import paho.mqtt.client as mqtt
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import json

MODEL_PATH = 'runs/detect/train/weights/best.pt'
MQTT_BROKER = 'nozomi.proxy.rlwy.net'  # Địa chỉ MQTT Broker
MQTT_PORT = 32067
REQUEST_TOPIC = "yolo/detect/request"
RESPONSE_TOPIC_PREFIX = "yolo/detect/response/"
# -----------------

try:
    model = YOLO(MODEL_PATH)
    print("Model YOLO đã được tải thành công!")
except Exception as e:
    print(f"Lỗi khi tải model: {e}")
    exit()

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("Server đã kết nối thành công đến MQTT Broker!")
        client.subscribe(REQUEST_TOPIC)
        print(f"Đang lắng nghe trên topic: {REQUEST_TOPIC}")
    else:
        print(f"Kết nối thất bại, mã lỗi: {rc}")

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        client_id = payload['client_id']
        image_b64 = payload['image_b64']
        
        # Tạo topic trả lời động
        response_topic = RESPONSE_TOPIC_PREFIX + client_id
        
        # Giải mã ảnh
        image_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Chạy nhận diện
        results = model(img)

        # Trích xuất kết quả và luôn trả về JSON
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[cls_id]
                detections.append({
                    "class_name": class_name,
                    "confidence": float(conf),
                    "box": [int(x1), int(y1), int(x2), int(y2)]
                })
        
        response_payload = json.dumps(detections)
        client.publish(response_topic, response_payload)
        
    except Exception as e:
        print(f"Lỗi khi xử lý tin nhắn: {e}")

# Khởi tạo và chạy MQTT client
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_forever()