"""Mengubah hasil unduhan Roboflow (format YOLO) menjadi struktur dataset_yolo/ untuk skrip 01, 05, 06.

Masukan  : folder hasil ekstrak zip Roboflow, berisi
           data.yaml, train/{images,labels}, valid/{images,labels}, test/{images,labels}
Keluaran : dataset_yolo/
           ├── classes.txt      # satu nama kelas per baris, urutan = class_id
           ├── images/
           └── labels/

Semua bagian (train/valid/test) dari Roboflow digabung lagi menjadi satu kumpulan,
karena pembagian train/val/test dilakukan oleh 05_split_dataset_yolo.py.

Pemakaian : python 00_siapkan_dari_roboflow.py [folder_roboflow] [folder_keluaran]
            (tanpa argumen: dataset_roboflow -> dataset_yolo)
Lanjutan  : python 01_cek_anotasi.py dataset_yolo
"""
import shutil
import sys
from pathlib import Path

import yaml

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "dataset_roboflow")
DST = Path(sys.argv[2] if len(sys.argv) > 2 else "dataset_yolo")
BAGIAN = ["train", "valid", "val", "test"]
EKSTENSI = (".jpg", ".jpeg", ".png")

if not (SRC / "data.yaml").exists():
    sys.exit(f"'{SRC / 'data.yaml'}' tidak ditemukan.\n"
             "Ekstrak zip Roboflow ke folder tersebut, atau tulis nama foldernya:\n"
             "  python 00_siapkan_dari_roboflow.py <folder_roboflow> [folder_keluaran]")
if DST.exists():
    sys.exit(f"Folder '{DST}' sudah ada. Skrip ini tidak menimpanya supaya data Anda aman.\n"
             "Hapus atau ganti nama folder itu, atau pilih nama keluaran lain.")

cfg = yaml.safe_load((SRC / "data.yaml").read_text(encoding="utf-8"))
names = cfg["names"]
kelas = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
print(f"Kelas dari data.yaml ({len(kelas)}): " + ", ".join(f"{i} = {k}" for i, k in enumerate(kelas)))

(DST / "images").mkdir(parents=True)
(DST / "labels").mkdir(parents=True)
(DST / "classes.txt").write_text("\n".join(kelas) + "\n", encoding="utf-8")

total, kosong, tanpa_label, di_luar = 0, 0, 0, 0
ditemukan = []
for bagian in BAGIAN:
    folder_img = SRC / bagian / "images"
    if not folder_img.exists():
        continue
    ditemukan.append(bagian)
    for img in sorted(p for p in folder_img.iterdir() if p.suffix.lower() in EKSTENSI):
        lbl = SRC / bagian / "labels" / f"{img.stem}.txt"
        nama = img.name
        if (DST / "images" / nama).exists():          # hindari nama kembar antar bagian
            nama = f"{bagian}_{nama}"
        stem = Path(nama).stem
        shutil.copy(img, DST / "images" / nama)
        if lbl.exists():
            isi = lbl.read_text()
            baris = [b for b in isi.splitlines() if b.strip()]
            if not baris:
                kosong += 1
            for b in baris:
                if int(b.split()[0]) >= len(kelas):
                    di_luar += 1
            (DST / "labels" / f"{stem}.txt").write_text(isi)
        else:                                          # tanpa file label = gambar latar
            tanpa_label += 1
            (DST / "labels" / f"{stem}.txt").write_text("")
        total += 1

print(f"Bagian Roboflow yang digabung : {', '.join(ditemukan) or '(tidak ada)'}")
print(f"Total gambar                  : {total}")
print(f"Gambar latar (label kosong)   : {kosong + tanpa_label}")
if di_luar:
    print(f"[!] {di_luar} kotak punya class_id di luar daftar kelas. Periksa data.yaml dan label.")
print(f"Siap di {DST}/ -> lanjutkan dengan: python 01_cek_anotasi.py {DST}")
