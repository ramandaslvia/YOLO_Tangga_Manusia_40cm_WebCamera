import os
import cv2

# folder foto dibuat otomatis di samping file skrip ini
FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "folder_papan_baru")
os.makedirs(FOLDER, exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # 0 = webcam baru
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
nomor = len(os.listdir(FOLDER))            # lanjut dari jumlah foto yang sudah ada

while True:
    ok, frame = cap.read()
    if not ok:
        print("Kamera tidak terbaca. Coba ganti angka 1 menjadi 0.")
        break

    cv2.imshow("kamera", frame)
    tombol = cv2.waitKey(1) & 0xFF

    if tombol == ord('s'):
        nomor += 1
        nama = os.path.join(FOLDER, f"foto_{nomor:02d}.jpg")
        if cv2.imwrite(nama, frame):
            print("tersimpan:", nama)
        else:
            print("GAGAL menyimpan:", nama)
    elif tombol == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()