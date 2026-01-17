from ultralytics import YOLO
import cv2

# 1. Panggil Model yang BARU saja lulus training
# Perhatikan path-nya: runs/detect/train2/weights/best.pt
model = YOLO('runs/detect/train2/weights/best.pt')

# 2. Nyalakan Kamera Laptop
cap = cv2.VideoCapture(0)

print("Kamera sedang dinyalakan... Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca kamera")
        break

    # 3. Suruh AI mendeteksi gambar dari kamera
    # conf=0.5 artinya cuma tampilkan kalau dia yakin di atas 50%
    results = model.predict(frame, conf=0.4, show=False)

    # 4. Gambar kotak hasil deteksi di layar
    annotated_frame = results[0].plot()

    # 5. Tampilkan jendela hasil
    cv2.imshow("TESTING NUTRI-SCANNER", annotated_frame)

    # Tekan tombol 'q' di keyboard untuk stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()