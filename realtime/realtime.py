"""Uji real-time: deteksi tangga (stair) dan manusia (person), lalu tandai tangga yang terhalang.

Letakkan file ini satu folder dengan best.pt (hasil training), lalu jalankan:
    python realtime.py
Tekan q di jendela video untuk keluar.
"""
import time

import cv2
from ultralytics import YOLO

MODEL = "best.pt"        # ganti jika nama atau lokasi file berbeda
KAMERA = 0               # nomor webcam (dari cek_kamera.py)
CONF = 0.4               # ambang keyakinan; turunkan ke 0.25 jika objek sering terlewat
IMGSZ = 640              # sama dengan ukuran latih model terpilih (E3_tuning)
AMBANG_TERHALANG = 0.3   # tangga dianggap terhalang jika >30% luasnya tertutup kotak manusia

model = YOLO(MODEL)
nama = {v: k for k, v in model.names.items()}
ID_TANGGA, ID_MANUSIA = nama["stair"], nama["person"]


def tumpang_tindih(t, m):
    """Porsi luas kotak tangga yang tertutup kotak manusia (0.0 sampai 1.0)."""
    x1, y1 = max(t[0], m[0]), max(t[1], m[1])
    x2, y2 = min(t[2], m[2]), min(t[3], m[3])
    luas = max(0, x2 - x1) * max(0, y2 - y1)
    luas_t = (t[2] - t[0]) * (t[3] - t[1])
    return luas / luas_t if luas_t > 0 else 0.0


cap = cv2.VideoCapture(KAMERA, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

t_awal = time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        print("Kamera tidak terbaca. Coba ganti KAMERA ke 1.")
        break

    hasil = model(frame, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
    tangga, manusia = [], []
    for b in hasil.boxes:
        kotak = b.xyxy[0].tolist()
        if int(b.cls) == ID_TANGGA:
            tangga.append(kotak)
        elif int(b.cls) == ID_MANUSIA:
            manusia.append(kotak)

    tampil = frame.copy()
    for m in manusia:
        cv2.rectangle(tampil, (int(m[0]), int(m[1])), (int(m[2]), int(m[3])), (255, 150, 0), 2)
        cv2.putText(tampil, "manusia", (int(m[0]), max(int(m[1]) - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)

    for t in tangga:
        persen = max([tumpang_tindih(t, m) for m in manusia], default=0.0)
        terhalang = persen > AMBANG_TERHALANG
        warna = (0, 0, 255) if terhalang else (0, 200, 0)
        label = f"tangga TERHALANG {persen:.0%}" if terhalang else "tangga"
        cv2.rectangle(tampil, (int(t[0]), int(t[1])), (int(t[2]), int(t[3])), warna, 2)
        cv2.putText(tampil, label, (int(t[0]), max(int(t[1]) - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, warna, 2)

    cv2.imshow("Deteksi tangga dan manusia (q = keluar)", tampil)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
