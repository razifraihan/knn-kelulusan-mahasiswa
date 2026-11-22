"""
Prediksi Kelulusan Mahasiswa dengan K-Nearest Neighbors (KNN) dari nol.

Dataset: 26 mahasiswa (soal UTS Data Mining).
Fitur  : IPS S3-S6 (numerik), Asal Sekolah (kategorikal), JK (kategorikal)
Label  : Lulus / Tidak Lulus

Hanya memakai pustaka standar Python, jadi seluruh logika algoritmanya
(jarak Euclidean, pengurutan tetangga, voting) terlihat jelas.
"""
import math
import random
from collections import Counter

DATA = [
    ("SMA", "P", 3.60, 3.58, 3.05, 3.70, "Lulus"),
    ("SMA", "L", 3.60, 3.68, 3.30, 3.85, "Lulus"),
    ("SMK", "L", 3.05, 3.05, 3.30, 3.40, "Lulus"),
    ("SMK KESEHATAN", "L", 3.30, 3.42, 3.55, 3.70, "Lulus"),
    ("SMA", "L", 3.05, 3.20, 3.20, 3.40, "Lulus"),
    ("SMA", "P", 3.35, 3.00, 2.85, 3.10, "Lulus"),
    ("MA", "L", 2.02, 2.03, 1.83, 0.85, "Tidak Lulus"),
    ("SMA", "L", 3.30, 2.79, 3.45, 3.55, "Lulus"),
    ("MA", "P", 3.05, 2.79, 3.00, 2.85, "Tidak Lulus"),
    ("MA", "P", 3.05, 2.95, 3.45, 3.25, "Tidak Lulus"),
    ("SMK", "L", 2.38, 1.58, 0.00, 1.43, "Tidak Lulus"),
    ("SMK", "L", 2.15, 2.02, 2.03, 1.83, "Tidak Lulus"),
    ("SMK KESEHATAN", "P", 3.28, 3.37, 3.54, 3.35, "Lulus"),
    ("SMA", "P", 2.77, 3.05, 3.08, 3.15, "Lulus"),
    ("SMK", "P", 3.28, 2.13, 1.68, 2.87, "Tidak Lulus"),
    ("MA", "L", 3.28, 3.37, 3.54, 3.35, "Lulus"),
    ("SMA", "P", 3.03, 2.71, 2.70, 2.61, "Tidak Lulus"),
    ("SMK KESEHATAN", "L", 1.14, 1.49, 0.87, 1.47, "Tidak Lulus"),
    ("SMK KESEHATAN", "P", 3.04, 2.21, 2.12, 2.86, "Lulus"),
    ("SMA", "P", 3.13, 3.26, 3.36, 3.53, "Lulus"),
    ("SMK KESEHATAN", "L", 3.05, 1.99, 2.46, 0.94, "Tidak Lulus"),
    ("SMK KESEHATAN", "P", 3.04, 2.21, 2.12, 2.86, "Lulus"),
    ("SMK", "L", 2.14, 2.15, 2.59, 2.57, "Tidak Lulus"),
    ("SMA", "P", 3.35, 3.00, 2.85, 3.10, "Lulus"),
    ("MA", "L", 1.40, 2.39, 2.51, 3.56, "Tidak Lulus"),
    ("SMA", "L", 2.56, 2.50, 0.64, 1.66, "Lulus"),
]

POSITIF = "Lulus"
SEKOLAH = sorted({r[0] for r in DATA})


# ---------------------------------------------------------------- fitur
def fit_scaler(rows):
    """Cari min/max tiap IPS dari data latih saja (hindari data leakage)."""
    kolom = list(zip(*[r[2:6] for r in rows]))
    return [(min(k), max(k)) for k in kolom]


def ke_vektor(row, scaler, pakai_kategori=True):
    """Min-max normalisasi IPS + one-hot Asal Sekolah + JK (L=0, P=1)."""
    vec = []
    for nilai, (lo, hi) in zip(row[2:6], scaler):
        vec.append((nilai - lo) / (hi - lo) if hi > lo else 0.0)
    if pakai_kategori:
        vec += [1.0 if row[0] == s else 0.0 for s in SEKOLAH]
        vec.append(1.0 if row[1] == "P" else 0.0)
    return vec


# ---------------------------------------------------------------- KNN
def euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def knn_predict(train_X, train_y, x, k):
    """Hitung jarak ke semua data latih, urutkan, ambil k terdekat, voting."""
    jarak = sorted((euclidean(x, t), y) for t, y in zip(train_X, train_y))
    tetangga = jarak[:k]
    suara = Counter(y for _, y in tetangga)
    teratas = suara.most_common()
    if len(teratas) > 1 and teratas[0][1] == teratas[1][1]:
        return tetangga[0][1]  # seri -> ikuti tetangga terdekat
    return teratas[0][0]


# ---------------------------------------------------------------- metrik
def metrik(y_true, y_pred):
    tp = sum(t == POSITIF and p == POSITIF for t, p in zip(y_true, y_pred))
    tn = sum(t != POSITIF and p != POSITIF for t, p in zip(y_true, y_pred))
    fp = sum(t != POSITIF and p == POSITIF for t, p in zip(y_true, y_pred))
    fn = sum(t == POSITIF and p != POSITIF for t, p in zip(y_true, y_pred))
    total = tp + tn + fp + fn
    acc = (tp + tn) / total
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, accuracy=acc, precision=prec, recall=rec)


def cetak(judul, m):
    print(f"\n{judul}")
    print("  Confusion matrix (positif = Lulus)")
    print("                  Pred Lulus   Pred Tidak")
    print(f"  Aktual Lulus     {m['tp']:>6}       {m['fn']:>6}")
    print(f"  Aktual Tidak     {m['fp']:>6}       {m['tn']:>6}")
    print(f"  Accuracy : {m['accuracy']:.2%}")
    print(f"  Precision: {m['precision']:.2%}")
    print(f"  Recall   : {m['recall']:.2%}")


# ---------------------------------------------------------------- evaluasi
def leave_one_out(data, k, pakai_kategori=True):
    """Tiap data diuji bergantian, sisanya jadi data latih."""
    y_true, y_pred = [], []
    for i, uji in enumerate(data):
        latih = data[:i] + data[i + 1:]
        scaler = fit_scaler(latih)
        X = [ke_vektor(r, scaler, pakai_kategori) for r in latih]
        y = [r[6] for r in latih]
        y_pred.append(knn_predict(X, y, ke_vektor(uji, scaler, pakai_kategori), k))
        y_true.append(uji[6])
    return metrik(y_true, y_pred)


def split_stratified(data, rasio_latih=0.7, seed=1992):
    rnd = random.Random(seed)
    latih, uji = [], []
    for label in ("Lulus", "Tidak Lulus"):
        grup = [r for r in data if r[6] == label]
        rnd.shuffle(grup)
        n = round(len(grup) * rasio_latih)
        latih += grup[:n]
        uji += grup[n:]
    return latih, uji


def hold_out(data, k, seed=1992, pakai_kategori=True):
    latih, uji = split_stratified(data, 0.7, seed)
    scaler = fit_scaler(latih)
    X = [ke_vektor(r, scaler, pakai_kategori) for r in latih]
    y = [r[6] for r in latih]
    pred = [knn_predict(X, y, ke_vektor(r, scaler, pakai_kategori), k) for r in uji]
    return metrik([r[6] for r in uji], pred), len(latih), len(uji)


def main():
    n_lulus = sum(r[6] == "Lulus" for r in DATA)
    print(f"Jumlah data: {len(DATA)} (Lulus={n_lulus}, Tidak Lulus={len(DATA) - n_lulus})")

    print("\n=== Mencari nilai k terbaik (Leave-One-Out CV, semua fitur) ===")
    print(" k | accuracy | precision | recall")
    for k in (1, 3, 5, 7, 9):
        m = leave_one_out(DATA, k)
        print(f"{k:>2} | {m['accuracy']:>8.2%} | {m['precision']:>9.2%} | {m['recall']:>6.2%}")

    print("\n=== Pengaruh fitur kategorikal (k=5, Leave-One-Out) ===")
    for nama, flag in (("IPS saja", False), ("IPS + Asal Sekolah + JK", True)):
        m = leave_one_out(DATA, 5, flag)
        print(f"{nama:<26} accuracy={m['accuracy']:.2%} precision={m['precision']:.2%} recall={m['recall']:.2%}")

    m = leave_one_out(DATA, 5)
    cetak("=== HASIL UTAMA: k=5, Leave-One-Out CV (26 data) ===", m)

    m, n_latih, n_uji = hold_out(DATA, 5)
    cetak(f"=== Pembanding: split 70/30 stratified, k=5 ({n_latih} latih, {n_uji} uji) ===", m)

    # Opsional: verifikasi dengan scikit-learn
    try:
        from sklearn.model_selection import LeaveOneOut, cross_val_predict
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import MinMaxScaler

        X = [ke_vektor(r, [(min(c), max(c)) for c in zip(*[q[2:6] for q in DATA])]) for r in DATA]
        y = [r[6] for r in DATA]
        model = make_pipeline(MinMaxScaler(), KNeighborsClassifier(n_neighbors=5))
        pred = cross_val_predict(model, X, y, cv=LeaveOneOut())
        ms = metrik(y, list(pred))
        print(f"\n[Cek scikit-learn, LOO k=5] accuracy={ms['accuracy']:.2%} "
              f"precision={ms['precision']:.2%} recall={ms['recall']:.2%}")
    except ImportError:
        print("\n(scikit-learn tidak terpasang, langkah verifikasi dilewati)")


if __name__ == "__main__":
    main()
