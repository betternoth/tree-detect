from ultralytics import YOLO

# ถ้าไฟล์ของคุณชื่อ treedetect.pt (custom out of Ultralytics), ให้ระบุชื่อไฟล์นั้น
model = YOLO("treedetect.pt")

# ถ้าอยากใช้ yolov8n, ให้ระบุ yolov8n.pt
# model = YOLO("yolov8n.pt")

model.export(format="onnx", imgsz=640, dynamic=True)
print("Export completed: treedetect.onnx")
