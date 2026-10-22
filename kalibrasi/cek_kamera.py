import cv2

for i in range(5):
    print(f"Mencoba index {i} ...", flush=True)
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)

    if cap.isOpened():
        ok, frame = cap.read()
        if ok:
            h, w = frame.shape[:2]
            print(f"Index {i}: TERBUKA, resolusi default {w} x {h}")
            cv2.imshow(f"Kamera index {i}", frame)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
        else:
            print(f"Index {i}: terbuka tetapi tidak ada gambar")
        cap.release()
    else:
        print(f"Index {i}: tidak tersedia")