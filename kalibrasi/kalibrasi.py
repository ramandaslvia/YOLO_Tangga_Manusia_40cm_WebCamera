import os
import numpy as np
from kalibrasi_utils import calibrate_images

SINI = os.path.dirname(os.path.abspath(__file__))
pola = os.path.join(SINI, "folder_papan_baru", "*.jpg")
# deteksi sudut di semua foto, lalu hitung parameter kamera
hasil = calibrate_images(pola, display=False)

print("Jumlah foto :", len(hasil.image_paths))
print("Terdeteksi  :", hasil.detected_images)
print("RMS (piksel):", round(hasil.rms_error, 3))
print("Matriks K:\n", hasil.camera_matrix)
print("Distorsi:\n", hasil.distortion_coefficients)

np.savez(
    os.path.join(SINI, "calib_baru.npz"),
    K=hasil.camera_matrix,
    dist=hasil.distortion_coefficients,
    image_size=np.array(hasil.image_size),
    rms=hasil.rms_error,
)
print("Tersimpan ke calib_baru.npz")