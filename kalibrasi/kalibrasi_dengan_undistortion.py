"""Membuat dokumentasi kalibrasi kamera dari folder_papan_baru dan calib_baru.npz.

Hasil (di folder 'dokumentasi'):
    1_deteksi_sudut/  foto papan dengan titik dan garis sudut yang terdeteksi
    2_sebelum/        foto asli dari kamera (masih ada distorsi)
    3_sesudah/        foto yang sama setelah dikoreksi (undistort)
    4_perbandingan/   sebelum dan sesudah dalam satu gambar
    ringkasan.txt     angka hasil kalibrasi
"""

import glob
import os

import cv2
import numpy as np

from kalibrasi_utils import CHECKERBOARD

SINI = os.path.dirname(os.path.abspath(__file__))
FOLDER_FOTO = os.path.join(SINI, "folder_papan_baru")
FOLDER_HASIL = os.path.join(SINI, "dokumentasi")
JUMLAH_PEMBANDING = 6      # berapa foto yang dibuat sebelum-sesudahnya

sub = {}
for nama in ["1_deteksi_sudut", "2_sebelum", "3_sesudah", "4_perbandingan"]:
    sub[nama] = os.path.join(FOLDER_HASIL, nama)
    os.makedirs(sub[nama], exist_ok=True)

# ---- ambil hasil kalibrasi ----
kalib = np.load(os.path.join(SINI, "calib_baru.npz"))
K = kalib["K"]
dist = kalib["dist"]
ukuran = (int(kalib["image_size"][0]), int(kalib["image_size"][1]))   # (lebar, tinggi)
rms = float(kalib["rms"])

# ---- bagian 1: gambar sudut yang terdeteksi ----
daftar_foto = sorted(glob.glob(os.path.join(FOLDER_FOTO, "*.jpg")))
berhasil = []
gagal = []
for jalur in daftar_foto:
    nama_file = os.path.basename(jalur)
    gambar = cv2.imread(jalur)
    abu = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
    ketemu, sudut = cv2.findChessboardCornersSB(
        abu, CHECKERBOARD, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    )
    if ketemu:
        cv2.drawChessboardCorners(gambar, CHECKERBOARD, sudut, ketemu)
        cv2.imwrite(os.path.join(sub["1_deteksi_sudut"], nama_file), gambar)
        berhasil.append(jalur)
    else:
        gagal.append(nama_file)
    print("deteksi:", nama_file, "OK" if ketemu else "GAGAL")

# ---- bagian 2: sebelum dan sesudah undistort ----
# alpha = 1: semua piksel asli dipertahankan, jadi tepi hitam terlihat (slide 30)
K_baru, _ = cv2.getOptimalNewCameraMatrix(K, dist, ukuran, 1, ukuran)

langkah = max(1, len(berhasil) // JUMLAH_PEMBANDING)
terpilih = berhasil[::langkah][:JUMLAH_PEMBANDING]
for jalur in terpilih:
    nama = os.path.splitext(os.path.basename(jalur))[0]
    sebelum = cv2.imread(jalur)
    sesudah = cv2.undistort(sebelum, K, dist, None, K_baru)

    cv2.imwrite(os.path.join(sub["2_sebelum"], nama + "_sebelum.jpg"), sebelum)
    cv2.imwrite(os.path.join(sub["3_sesudah"], nama + "_sesudah.jpg"), sesudah)

    kiri = sebelum.copy()
    kanan = sesudah.copy()
    cv2.putText(kiri, "SEBELUM (distorted)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(kanan, "SESUDAH (undistorted)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
    cv2.imwrite(os.path.join(sub["4_perbandingan"], nama + "_perbandingan.jpg"),
                np.hstack([kiri, kanan]))

# ---- bagian 3: ringkasan angka ----
d = dist.ravel()
baris = [
    "HASIL KALIBRASI KAMERA",
    "======================",
    f"Resolusi foto        : {ukuran[0]} x {ukuran[1]}",
    f"Pola papan           : {CHECKERBOARD[0]} x {CHECKERBOARD[1]} sudut dalam",
    f"Jumlah foto          : {len(daftar_foto)}",
    f"Terdeteksi           : {len(berhasil)}",
    f"Gagal                : {len(gagal)}",
    f"RMS reprojection     : {rms:.3f} piksel",
    "",
    "Matriks kamera K:",
    np.array2string(K, precision=2, suppress_small=True),
    f"fx = {K[0, 0]:.2f}   fy = {K[1, 1]:.2f}",
    f"cx = {K[0, 2]:.2f}   cy = {K[1, 2]:.2f}",
    "",
    "Koefisien distorsi (k1, k2, p1, p2, k3):",
    ", ".join(f"{v:.5f}" for v in d[:5]),
    "",
    "Perubahan dari kode asli (calibration_utils.py):",
    "1. CHECKERBOARD diubah dari (6, 9) menjadi (7, 7)",
    "2. Detektor diganti ke findChessboardCornersSB",
    "3. k3 dikunci = 0 dengan flag CALIB_FIX_K3",
    "",
    "Foto yang gagal dideteksi:",
]
baris += gagal if gagal else ["(tidak ada)"]
with open(os.path.join(FOLDER_HASIL, "ringkasan.txt"), "w") as f:
    f.write("\n".join(baris) + "\n")

print()
print("Terdeteksi", len(berhasil), "dari", len(daftar_foto), "foto")
print("Foto pembanding:", len(terpilih))
print("Dokumentasi tersimpan di:", FOLDER_HASIL)