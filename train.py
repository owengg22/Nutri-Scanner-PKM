from ultralytics import YOLO

# 1. Pilih Model: Kita pakai 'yolov8n.pt' (Nano)
# Kenapa Nano? Karena ini paling ringan & cepat buat Raspberry Pi 5 kamu!
model = YOLO('yolov8n.pt')

# 2. Mulai Latihan (Training)
# epochs=50 artinya dia bakal baca buku pelajarannya diulang 50 kali (biar pinter).
# imgsz=640 adalah ukuran gambarnya.
if __name__ == '__main__':
    model.train(data='Nutri-Scanner-PKM-1/data.yaml', epochs=50, imgsz=640)