from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8n.pt")  # ?먮뒗 "yolo11n.pt"

    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        project="runs/detect",
        name="custom_yolo"
    )
