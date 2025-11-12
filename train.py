if __name__ == "__main__":
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    model.train(
        data="./data.yaml",
        batch=16,
        epochs=15,
        imgsz=768,
        device=0  
    )
