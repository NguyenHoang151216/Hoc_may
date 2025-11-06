from ultralytics import YOLO
import cv2

# 1️⃣ Load mô hình YOLO
model = YOLO("D:\\Hocmay\\yolo_results_1\\arrow_detection\\weights\\best.pt")

# 2️⃣ Đường dẫn video đầu vào
video_path = "D:\\Hocmay\\7170714254787.mp4"  # đổi thành video của bạn
cap = cv2.VideoCapture(video_path)

# 3️⃣ Cấu hình output video (nếu muốn lưu kết quả)
output_path = "result_with_tracking.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 4️⃣ Kẻ vạch cố định
start_point = (250, 270)
end_point = (650, 265)
color_line = (0, 255, 255)
thickness = 3

# 5️⃣ Chạy tracking
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Thực hiện detect + track
    results = model.track(source=frame, persist=True, conf=0.3, show=False)

    # Vẽ bounding box + tracking ID lên khung hình
    annotated_frame = results[0].plot()

    # Vẽ vạch cố định
    cv2.line(annotated_frame, start_point, end_point, color_line, thickness)

    # Hiển thị kết quả
    cv2.imshow("YOLOv8 Tracking", annotated_frame)

    # Ghi vào video output
    out.write(annotated_frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 6️⃣ Giải phóng tài nguyên
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Hoàn tất! Video kết quả lưu tại:", output_path)
