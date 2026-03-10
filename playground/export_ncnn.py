from ultralytics import YOLO

model = YOLO("models/vueltaalpartido_v1/best.pt")
model.export(format="ncnn", imgsz=640)
