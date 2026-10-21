"""ฟังก์ชันสำหรับสร้างข้อมูลตัวอย่างและแบ่งกลุ่มลูกค้าด้วย K-Means
(แยกจาก app.py เพื่อให้ทดสอบตรรกะได้โดยไม่ต้องเปิดเว็บ)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SAMPLE_FEATURES = [
    "Age",
    "Annual_Income_kTHB",
    "Spending_Score",
    "Visits_per_Month",
]


def make_sample_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """สร้างข้อมูลลูกค้าจำลอง 4 กลุ่มแฝง"""
    rng = np.random.default_rng(seed)
    # (สัดส่วน, อายุ, รายได้ (พันบาท/ปี), คะแนนการใช้จ่าย, จำนวนครั้งที่มาซื้อ/เดือน)
    groups = [
        (0.25, (24, 3), (35, 6), (78, 8), (9, 2)),    # วัยรุ่นใช้จ่ายสูง
        (0.30, (45, 6), (85, 10), (75, 9), (6, 1.5)),  # รายได้สูง ใช้จ่ายสูง
        (0.25, (50, 7), (80, 10), (25, 8), (2, 1)),    # รายได้สูง ประหยัด
        (0.20, (35, 8), (30, 6), (30, 8), (3, 1)),     # รายได้น้อย ใช้จ่ายน้อย
    ]
    frames = []
    for share, age, income, spend, visits in groups:
        size = int(n * share)
        frames.append(
            pd.DataFrame(
                {
                    "Age": rng.normal(*age, size),
                    "Annual_Income_kTHB": rng.normal(*income, size),
                    "Spending_Score": rng.normal(*spend, size),
                    "Visits_per_Month": rng.normal(*visits, size),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    df["Age"] = df["Age"].clip(18, 70).round().astype(int)
    df["Annual_Income_kTHB"] = df["Annual_Income_kTHB"].clip(10, 200).round(1)
    df["Spending_Score"] = df["Spending_Score"].clip(1, 100).round().astype(int)
    df["Visits_per_Month"] = df["Visits_per_Month"].clip(0, 20).round().astype(int)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "CustomerID", np.arange(1, len(df) + 1))
    return df


def run_kmeans(
    X: pd.DataFrame, k: int, seed: int = 42
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """สเกลข้อมูลแล้วทำ K-Means

    คืนค่า: (labels, silhouette, scaled_X, centers_ในสเกลเดิม)
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = model.fit_predict(Xs)
    score = float(silhouette_score(Xs, labels)) if k > 1 else float("nan")
    centers = scaler.inverse_transform(model.cluster_centers_)
    return labels, score, Xs, centers


def evaluate_k_range(
    X: pd.DataFrame, k_min: int = 2, k_max: int = 10, seed: int = 42
) -> pd.DataFrame:
    """คำนวณ Inertia (Elbow) และ Silhouette ของแต่ละค่า k"""
    Xs = StandardScaler().fit_transform(X)
    k_max = min(k_max, len(Xs) - 1)
    rows = []
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = model.fit_predict(Xs)
        rows.append(
            {
                "k": k,
                "Inertia": float(model.inertia_),
                "Silhouette": float(silhouette_score(Xs, labels)),
            }
        )
    return pd.DataFrame(rows)


def pca_2d(Xs: np.ndarray, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """ลดมิติเหลือ 2 แกนเพื่อพล็อต คืนค่า (พิกัด, สัดส่วนความแปรปรวนที่อธิบายได้)"""
    n_comp = min(2, Xs.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed)
    coords = pca.fit_transform(Xs)
    if n_comp == 1:  # กันกรณีมีฟีเจอร์เดียว
        coords = np.column_stack([coords[:, 0], np.zeros(len(coords))])
        return coords, np.array([pca.explained_variance_ratio_[0], 0.0])
    return coords, pca.explained_variance_ratio_


def cluster_profile(df: pd.DataFrame, features: list[str], label_col: str = "Cluster") -> pd.DataFrame:
    """ตารางสรุปค่าเฉลี่ยของแต่ละกลุ่ม พร้อมจำนวนและสัดส่วน"""
    prof = df.groupby(label_col)[features].mean().round(2)
    counts = df[label_col].value_counts().sort_index()
    prof.insert(0, "จำนวนลูกค้า", counts)
    prof.insert(1, "สัดส่วน (%)", (counts / counts.sum() * 100).round(1))
    return prof


def describe_clusters(df: pd.DataFrame, features: list[str], label_col: str = "Cluster", threshold: float = 0.5) -> dict[int, str]:
    """อธิบายลักษณะแต่ละกลุ่มโดยเทียบกับค่าเฉลี่ยรวม (หน่วยเป็นส่วนเบี่ยงเบนมาตรฐาน)"""
    overall_mean = df[features].mean()
    overall_std = df[features].std().replace(0, 1)
    result: dict[int, str] = {}
    for c, g in df.groupby(label_col):
        z = (g[features].mean() - overall_mean) / overall_std
        high = [f for f in features if z[f] >= threshold]
        low = [f for f in features if z[f] <= -threshold]
        parts = []
        if high:
            parts.append("สูงกว่าค่าเฉลี่ย: " + ", ".join(high))
        if low:
            parts.append("ต่ำกว่าค่าเฉลี่ย: " + ", ".join(low))
        result[int(c)] = " | ".join(parts) if parts else "ใกล้เคียงค่าเฉลี่ยทุกตัวแปร"
    return result
