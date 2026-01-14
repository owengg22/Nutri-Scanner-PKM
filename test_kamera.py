from ultralytics import YOLO

# 1. Load model bawaan (YOLOv8 Nano)
# Karena belum punya file sendiri, dia akan download otomatis dari internet
print("loading model...")
model = YOLO('yolov8n.pt') 

print("Menyalakan kamera")
results = model.predict(source='0', show=True, conf=0.3)