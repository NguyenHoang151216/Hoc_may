from ultralytics import YOLO
import cv2
import os
import datetime
import numpy as np
from shapely.geometry import Point, Polygon

# ==== Load model YOLO ====
model = YOLO("D:\\Hocmay\\yolo_results_1\\arrow_detection\\weights\\best.pt")

# ==== Video input / output ====
video_path = "D:\\Hocmay\\7215132097540.mp4"
cap = cv2.VideoCapture(video_path)

output_path = "result_violation_tracking.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# ==== Folder lưu ảnh vi phạm ====
save_dir = "violations"
os.makedirs(save_dir, exist_ok=True)

# --- Vùng vi phạm (vùng đi thẳng) ---
violation_zone_points = np.array([
    [380, 110],
    [560, 110],
    [640, 270],
    [260, 275]
])
polygon_violation = Polygon(violation_zone_points)

# Vùng làn phải (được phép rẽ phải khi đèn đỏ)
right_lane_points = np.array([
    [520, 160],
    [640, 160],
    [width - 50, height],
    [600, height]
])
polygon_right_lane = Polygon(right_lane_points)

# ==== Biến trạng thái ====
current_light = "unknown"
previous_light = "unknown"
red_start_time = None

tracked_vehicles = {}

print("Đang chạy phát hiện vi phạm.Nhấn 'q' để thoát.")

tracker = model.track(source=video_path, conf=0.3, show=False, persist=True, stream=True)

for results in tracker:
    annotated = results.plot()
    frame = results.orig_img
    names = results.names

    boxes = results.boxes.xyxy.cpu().numpy()
    cls_ids = results.boxes.cls.cpu().numpy().astype(int)
    ids = results.boxes.id.cpu().numpy().astype(int) if results.boxes.id is not None else []

    # === Xác định trạng thái đèn ===
    previous_light = current_light
    new_light_state = None

    for cls_id in cls_ids:
        label = names[cls_id]
        if label == "denxanh":
            new_light_state = "green"
            break
        elif label == "dendo":
            new_light_state = "red"

    if new_light_state:
        current_light = new_light_state

    # Ghi nhận thời điểm bắt đầu đèn đỏ
    if current_light == "red" and previous_light != "red":
        red_start_time = datetime.datetime.now()
        print("Đèn chuyển sang ĐỎ tại:", red_start_time.strftime("%H:%M:%S"))

    # === Vẽ vùng ===
    cv2.polylines(annotated, [violation_zone_points], True, (0, 255, 255), 2)
    cv2.polylines(annotated, [right_lane_points], True, (0, 200, 0), 2)

    # === Xử lý từng xe ===
    for box, cls_id, track_id in zip(boxes, cls_ids, ids):
        label = names[cls_id]
        if label not in ["oto", "xemay"]:
            continue

        x1, y1, x2, y2 = map(int, box)
        center_x = (x1 + x2) // 2
        bottom_y = y2
        bottom_point = Point(center_x, bottom_y)
        color = (0, 255, 0)

        if track_id not in tracked_vehicles:
            tracked_vehicles[track_id] = {
                "entered_before_red": (current_light != "red"),
                "violated": False,
                "history": []
            }

        tracked_vehicles[track_id]["history"].append((center_x, bottom_y))
        if len(tracked_vehicles[track_id]["history"]) > 15:
            tracked_vehicles[track_id]["history"].pop(0)

        entered_before_red = tracked_vehicles[track_id]["entered_before_red"]

        # ====== Xác định hướng di chuyển ======
        direction = "unknown"
        history = tracked_vehicles[track_id]["history"]
        if len(history) >= 2:
            dx = history[-1][0] - history[0][0]
            dy = history[-1][1] - history[0][1]
            if abs(dy) > abs(dx):
                if dy < 0:
                    direction = "up"
                else:
                    direction = "down"
            else:
                if dx > 0:
                    direction = "right"
                else:
                    direction = "left"

        # ====== Logic vi phạm ======
        if current_light == "red" and red_start_time and not tracked_vehicles[track_id]["violated"]:
            in_right_lane_now = polygon_right_lane.contains(bottom_point)
            in_violation_zone_now = polygon_violation.contains(bottom_point)
            in_right_lane_before = any(polygon_right_lane.contains(Point(x, y)) for x, y in history[:-2])

            if direction == "up":
                if in_violation_zone_now and not entered_before_red:
                    if not in_right_lane_now and not in_right_lane_before:
                        tracked_vehicles[track_id]["violated"] = True
                        color = (0, 0, 255)
                        filename = os.path.join(
                            save_dir,
                            f"violation_{label}_{track_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f')}.jpg"
                        )
                        cv2.imwrite(filename, frame[y1:y2, x1:x2])
                        print(f"🚨 Vi phạm vượt đèn đỏ: {label}, ID={track_id}, lưu tại {filename}")

        if tracked_vehicles[track_id]["violated"]:
            color = (0, 0, 255)

        # Vẽ khung xe
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        # Nếu muốn bật label thì bỏ chú thích ở dòng dưới:
        # cv2.putText(annotated, f"{label} #{track_id} ({direction})", (x1, y1 - 5),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # === Hiển thị trạng thái đèn ===
    cv2.putText(
        annotated,
        f"LIGHT: {current_light.upper()}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255) if current_light == "red" else (0, 255, 0),
        3
    )

    out.write(annotated)
    cv2.imshow("Traffic Violation Detection (Tracking)", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Hoàn tất! Video kết quả lưu tại:", output_path)
