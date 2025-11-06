from ultralytics import YOLO
import cv2
import os
import datetime
import numpy as np
from shapely.geometry import Point, Polygon

#Load mô hình YOLO
model = YOLO("D:\\Hocmay\\yolo_results_1\\arrow_detection\\weights\\best.pt")

#Video đầu vào
video_path = "D:\\Hocmay\\7170649511227.mp4"
cap = cv2.VideoCapture(video_path)

#Video đầu ra
output_path = "result_violation_polygon.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#Tạo folder lưu vi phạm
save_dir = "violations"
os.makedirs(save_dir, exist_ok=True)


violation_zone_points = np.array([
    [400, 100],
    [550, 100],
    [630, 235],   
    [270, 240]
    
])
polygon_zone = Polygon(violation_zone_points)

#Vẽ vạch (chỉ để hiển thị)
start_point = (250, 270)
end_point = (650, 265)
color_line = (0, 255, 255)
thickness = 3

#Biến trạng thái
current_light = "unknown"

print("🚦 Đang chạy phát hiện vi phạm... Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.25)
    annotated_frame = frame.copy()

    boxes = results[0].boxes.xyxy.cpu().numpy()
    cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    names = results[0].names

    #Xác định trạng thái đèn
    current_light = "unknown"
    for cls_id, box in zip(cls_ids, boxes):
        label = names[cls_id]
        if label == "denxanh":
            current_light = "green"
            break
        elif label == "dendo":
            current_light = "red"

    #Vẽ vùng polygon (vùng vi phạm)
    cv2.polylines(annotated_frame, [violation_zone_points], True, (0, 255, 255), 2)

    #Xử lý phương tiện
    for cls_id, box in zip(cls_ids, boxes):
        label = names[cls_id]
        if label not in ["oto", "xemay"]:
            continue

        x1, y1, x2, y2 = map(int, box)
        center_x = (x1 + x2) // 2
        bottom_y = y2
        center_bottom = Point(center_x, bottom_y)
        color = (0, 255, 0)  # mặc định xanh (hợp lệ)

        if current_light == "red":
            # Xe rẽ phải hợp lệ (phần 1/4 phải màn hình)
            if center_x > width * 0.75:
                color = (255, 255, 0) 
            # Xe vượt vạch thật sự (lọt vào polygon vi phạm)
            elif polygon_zone.contains(center_bottom):
                color = (0, 0, 255)

                # Lưu ảnh xe vi phạm
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
                filename = f"{save_dir}/violation_{label}_{timestamp}.jpg"
                cv2.imwrite(filename, frame[y1:y2, x1:x2])
                print(f"🚨 Vi phạm: {label} vượt đèn đỏ, lưu tại {filename}")

        # Vẽ bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    #Hiển thị trạng thái đèn
    cv2.putText(annotated_frame, f"LIGHT: {current_light.upper()}",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 0, 255) if current_light == "red" else (0, 255, 0), 3)

    # Hiển thị và ghi video
    cv2.imshow("Traffic Violation Detection", annotated_frame)
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Giải phóng tài nguyên
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Hoàn tất! Video kết quả lưu tại:", output_path)
