import cv2
from ultralytics import YOLO

# URL stream RTSP
rtsp_url = "rtsp://freja.hiof.no:1935/rtplive/_definst_/hessdalen02.stream"

# Load model YOLO (pastikan model sudah terunduh, misal: 'yolov8n.pt')
model = YOLO('yolov8n.pt')  # Gunakan model YOLOv8 yang sesuai

# Fungsi untuk mendeteksi mobil
def detect_vehicle(frame):
    results = model.predict(source=frame, conf=0.5, classes=[2, 3, 5, 7])  # Deteksi mobil, bus, truk, dan motor
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Koordinat bounding box
            confidence = box.conf[0]  # Kepercayaan deteksi
            class_id = int(box.cls[0])  # ID kelas
            label = model.names[class_id]  # Label kelas

            # Gambar kotak deteksi dan label pada frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Membuka stream kamera
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Tidak dapat membuka stream RTSP")
    exit()

print("Memulai deteksi kendaraan...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Tidak dapat membaca frame dari stream.")
        break

    # Resize frame agar lebih cepat diproses (opsional)
    frame_resized = cv2.resize(frame, (640, 480))

    # Deteksi kendaraan pada frame
    frame_with_detections = detect_vehicle(frame_resized)

    # Tampilkan hasil deteksi
    cv2.imshow("Deteksi Kendaraan", frame_with_detections)

    # Keluar jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan sumber daya
cap.release()
cv2.destroyAllWindows()
