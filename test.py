from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# 1️⃣ Tải mô hình YOLO
model = YOLO("D:\\Hocmay\\yolo_results\\arrow_detection\\weights\\best.pt")

# 2️⃣ Đọc ảnh cần nhận diện
image_path = "D:\\Hocmay\\z7146913144441_79137576dda2a101e99a064da1f71480.jpg"

# 3️⃣ Thực hiện dự đoán
results = model.predict(source=image_path, conf=0.25)

# 4️⃣ Lấy ảnh có kết quả nhận diện (YOLO tự vẽ box)
annotated_img = results[0].plot()

# 5️⃣ KẺ VẠCH CỐ ĐỊNH
# --- ví dụ: kẻ 1 vạch ngang giữa ảnh (vạch dừng đèn đỏ)
height, width, _ = annotated_img.shape

# Bạn có thể thay đổi toạ độ tuỳ theo ảnh:
start_point = (326,365)   # điểm đầu
end_point = (805,357)     # điểm cuối
color = (0, 255, 255)   # màu vàng (BGR)
thickness = 3

# Vẽ vạch
cv2.line(annotated_img, start_point, end_point, color, thickness)

# 6️⃣ Hiển thị kết quả
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# 7️⃣ (tuỳ chọn) Lưu lại ảnh kết quả
cv2.imwrite("result_with_line.jpg", annotated_img)
print("✅ Ảnh kết quả đã lưu: result_with_line.jpg")