import json

# Đường dẫn tới file export Label Studio (ví dụ file bạn đã export ra)
input_file = "D:\Hocmay\project-10-at-2025-10-16-23-29-b17b0265.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

tasks_with_vachngang = []

for task in data:
    if "annotations" in task:
        for ann in task["annotations"]:
            if "result" in ann:
                for r in ann["result"]:
                    # Kiểm tra nhãn "vachngang" trong rectanglelabels
                    if "rectanglelabels" in r.get("value", {}):
                        if "traffic_light" in r["value"]["rectanglelabels"]:
                            tasks_with_vachngang.append({
                                "id": task.get("id"),
                                "image": task["data"].get("image")
                            })

print("🟡 Các ảnh còn chứa nhãn 'vachngang':")
for t in tasks_with_vachngang:
    print(f" - ID: {t['id']} | Ảnh: {t['image']}")

print(f"\nTổng cộng: {len(tasks_with_vachngang)} ảnh chứa nhãn 'vachngang'")
