from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data=r"Vueltaalpartido_v1\data.yaml",
        epochs=200,
        imgsz=640,
        batch=8,
        workers=0  # importante en Windows para evitar spawn issues
    )

if __name__ == "__main__":
    main()
