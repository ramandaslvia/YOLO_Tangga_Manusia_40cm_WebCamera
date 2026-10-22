import cv2, os

FOLDER = "dataset_mentah/sesi2_terang"
os.makedirs(FOLDER, exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
nomor = len(os.listdir(FOLDER))
print("Tekan s = simpan foto, q = keluar")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Kamera tidak terbaca")
        break

    tampil = frame.copy()
    cv2.putText(tampil, f"foto: {nomor}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("dataset", tampil)

    tombol = cv2.waitKey(1) & 0xFF
    if tombol == ord('s'):
        nomor += 1
        cv2.imwrite(f"{FOLDER}/img_{nomor:04d}.jpg", frame)   # simpan tanpa tulisan
    elif tombol == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()