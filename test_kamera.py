from ultralytics import YOLO
import cv2

# 1. Panggil Model
model = YOLO('runs/detect/train2/weights/best.pt')

# 2. Setting Kamera (Pilih 0, 1, atau 2)
source_kamera = 2 
cap = cv2.VideoCapture(source_kamera)

if not cap.isOpened():
    print("❌ ERROR: Kamera tidak terdeteksi!")
    exit()

print("✅ Kamera start! Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 3. Prediksi (Tanpa plot otomatis)
    results = model.predict(frame, conf=0.4, show=False)
    
    # 4. GAMBAR MANUAL (Supaya Namanya Bisa Diatur Suka-Suka)
    # Kita bongkar hasil deteksinya satu per satu
    for box in results[0].boxes:
        # Ambil koordinat kotak (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Ambil angka confidence (keyakinan)
        conf = float(box.conf[0])
        
        # Ambil nama asli dari model
        cls_id = int(box.cls[0])
        nama_asli = model.names[cls_id]

        # --- LOGIKA GANTI NAMA ---
        # Kalau nama aslinya aneh-aneh ada bau 'banana', kita ganti jadi 'Banana'
        if 'banana' in nama_asli.lower():
            label_teks = f"Banana {conf:.2f}"
        else:
            # Kalau bukan pisang (misal jahe), biarin nama aslinya
            label_teks = f"{nama_asli} {conf:.2f}"
            
        # Gambar Kotak (Warna Hijau: 0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Tulisan Label di atas kotak
        cv2.putText(frame, label_teks, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 5. Tampilkan
    cv2.imshow("TESTING NUTRI-SCANNER", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


