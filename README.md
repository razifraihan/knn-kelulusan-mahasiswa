# Klasifikasi Kelulusan Mahasiswa dengan K-Nearest Neighbors (dari nol)

Implementasi algoritma KNN dengan Python murni (tanpa library ML) untuk
memprediksi apakah mahasiswa lulus tepat waktu berdasarkan IPS semester 3-6,
asal sekolah, dan jenis kelamin. Dataset berasal dari soal UTS Data Mining
(26 data).

## Cara kerja
1. Normalisasi IPS dengan min-max (dihitung dari data latih saja)
2. One-hot encoding untuk asal sekolah, encoding biner untuk jenis kelamin
3. Hitung jarak Euclidean ke seluruh data latih
4. Urutkan, ambil k tetangga terdekat, lalu voting (seri mengikuti tetangga terdekat)

## Cara menjalankan
python knn_kelulusan.py

Tidak perlu install apa pun. scikit-learn bersifat opsional, hanya untuk verifikasi.

## Hasil (k=5, Leave-One-Out CV, 26 data)
| Metrik | Nilai |
|---|---|
| Accuracy | 80,77% |
| Precision (Lulus) | 81,25% |
| Recall (Lulus) | 86,67% |

Hasil identik dengan KNeighborsClassifier milik scikit-learn.

## Temuan
- Nilai k terbaik: k=1 (84,62%), namun k=5 dipilih karena lebih stabil.
- Menambah fitur asal sekolah dan jenis kelamin menaikkan accuracy dari 69,23% ke 80,77%.

## Keterbatasan
Dataset hanya 26 data, sehingga hasil ini bersifat latihan dan tidak cukup
untuk generalisasi.

## Teknologi
Python 3 (pustaka standar: math, random, collections)