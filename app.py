import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from segmentation import (
    cluster_profile,
    describe_clusters,
    evaluate_k_range,
    make_sample_data,
    pca_2d,
    run_kmeans,
)

# ---------------------------------------------------------------- ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Customer Segmentation", page_icon="👥", layout="wide")

st.markdown(
    """
<style>
.stApp { background: #0b0e14; color: #eef2f7; }
[data-testid="stSidebar"] { background: #141923; border-right: 1px solid #262d3a; }
.block-container { padding-top: 2rem; }
h1, h2, h3 { color: #f5f7fb; }
div[data-testid="stMetric"] { background: #151b26; border: 1px solid #293142; padding: 16px; border-radius: 12px; }
div[data-testid="stDataFrame"] { border: 1px solid #293142; border-radius: 8px; }
</style>
""",
    unsafe_allow_html=True,
)

TEMPLATE = "plotly_dark"


# ---------------------------------------------------------------- ฟังก์ชันช่วย (มี cache)
@st.cache_data
def load_sample() -> pd.DataFrame:
    return make_sample_data()


@st.cache_data
def cached_kmeans(X: pd.DataFrame, k: int, seed: int):
    return run_kmeans(X, k, seed)


@st.cache_data
def cached_eval(X: pd.DataFrame, seed: int) -> pd.DataFrame:
    return evaluate_k_range(X, seed=seed)


def read_csv_safely(uploaded) -> pd.DataFrame:
    """ลองอ่าน CSV ด้วย encoding ที่พบบ่อย (รวมไฟล์ภาษาไทยจาก Excel)"""
    last_error = None
    for enc in ("utf-8-sig", "cp874", "latin-1"):
        try:
            uploaded.seek(0)
            return pd.read_csv(uploaded, encoding=enc)
        except Exception as e:  # noqa: BLE001
            last_error = e
    raise last_error


def silhouette_comment(score: float) -> str:
    if score >= 0.5:
        return "กลุ่มแยกจากกันชัดเจน (ดี)"
    if score >= 0.25:
        return "แยกกลุ่มได้พอใช้"
    return "กลุ่มทับซ้อนกันมาก (ควรลองเปลี่ยนจำนวนกลุ่มหรือตัวแปร)"


# ---------------------------------------------------------------- Sidebar
st.sidebar.title("⚙️ ตั้งค่า")
source = st.sidebar.radio("แหล่งข้อมูล", ["ข้อมูลตัวอย่าง", "อัปโหลดไฟล์ CSV"])

if source == "ข้อมูลตัวอย่าง":
    df = load_sample()
else:
    uploaded = st.sidebar.file_uploader("เลือกไฟล์ CSV", type=["csv"])
    if uploaded is None:
        st.title("👥 Customer Segmentation")
        st.info("กรุณาอัปโหลดไฟล์ CSV ที่แถบด้านซ้าย หรือเปลี่ยนไปใช้ข้อมูลตัวอย่าง")
        st.stop()
    try:
        df = read_csv_safely(uploaded)
    except Exception as e:  # noqa: BLE001
        st.error(f"อ่านไฟล์ไม่สำเร็จ: {e}")
        st.stop()

numeric_cols = df.select_dtypes("number").columns.tolist()
if len(numeric_cols) < 2:
    st.title("👥 Customer Segmentation")
    st.error("ข้อมูลต้องมีคอลัมน์ตัวเลขอย่างน้อย 2 คอลัมน์")
    st.stop()

default_feats = [c for c in numeric_cols if "id" not in c.lower()][:4] or numeric_cols[:4]
features = st.sidebar.multiselect("ตัวแปรที่ใช้แบ่งกลุ่ม (เลือกอย่างน้อย 2)", numeric_cols, default=default_feats)
if len(features) < 2:
    st.title("👥 Customer Segmentation")
    st.warning("กรุณาเลือกตัวแปรอย่างน้อย 2 ตัวที่แถบด้านซ้าย")
    st.stop()

work = df.dropna(subset=features).reset_index(drop=True)
dropped = len(df) - len(work)
if len(work) < 10:
    st.title("👥 Customer Segmentation")
    st.error("ข้อมูลที่ใช้ได้มีน้อยกว่า 10 แถว ไม่เพียงพอสำหรับการแบ่งกลุ่ม")
    st.stop()

max_k = min(10, len(work) - 1)
k = st.sidebar.slider("จำนวนกลุ่ม (K)", 2, max_k, min(4, max_k))
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

# ---------------------------------------------------------------- ประมวลผล
X = work[features]
labels, sil, Xs, centers = cached_kmeans(X, k, int(seed))
result = work.copy()
result["Cluster"] = labels
plot_df = result.assign(Cluster=result["Cluster"].astype(str))
cluster_order = {"Cluster": [str(i) for i in range(k)]}

# ---------------------------------------------------------------- หัวเว็บ
st.title("👥 Customer Segmentation")
st.caption("แบ่งกลุ่มลูกค้าด้วย K-Means Clustering · Streamlit + scikit-learn + Plotly")

if dropped:
    st.warning(f"ตัดแถวที่มีค่าว่างในตัวแปรที่เลือกออก {dropped:,} แถว")

largest_share = result["Cluster"].value_counts(normalize=True).max() * 100
c1, c2, c3, c4 = st.columns(4)
c1.metric("จำนวนลูกค้า", f"{len(result):,}")
c2.metric("จำนวนกลุ่ม (K)", k)
c3.metric("Silhouette Score", f"{sil:.3f}", help="ยิ่งใกล้ 1 ยิ่งแยกกลุ่มชัด")
c4.metric("กลุ่มที่ใหญ่ที่สุด", f"{largest_share:.1f}%")
st.caption(f"ผลประเมิน: {silhouette_comment(sil)}")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 ภาพรวมข้อมูล", "🎯 ผลการแบ่งกลุ่ม", "📈 เลือกจำนวนกลุ่มที่เหมาะสม", "📥 ข้อมูลผลลัพธ์"]
)

# ---------------------------------------------------------------- Tab 1: ภาพรวมข้อมูล
with tab1:
    st.subheader("ตัวอย่างข้อมูล (50 แถวแรก)")
    st.dataframe(df.head(50))

    st.subheader("สถิติเชิงพรรณนา")
    st.dataframe(work[features].describe().round(2))

    st.subheader("การกระจายของตัวแปร")
    feat_hist = st.selectbox("เลือกตัวแปร", features, key="hist_feat")
    fig_hist = px.histogram(work, x=feat_hist, nbins=30, template=TEMPLATE)
    st.plotly_chart(fig_hist)

    st.subheader("ความสัมพันธ์ระหว่างตัวแปร (Correlation)")
    corr = work[features].corr().round(2)
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, template=TEMPLATE)
    st.plotly_chart(fig_corr)

# ---------------------------------------------------------------- Tab 2: ผลการแบ่งกลุ่ม
with tab2:
    st.subheader("แผนภาพกลุ่มลูกค้า")
    views = ["PCA 2D (ลดมิติ)", "เลือกแกนเอง 2D"]
    if len(features) >= 3:
        views.append("3D")
    view = st.radio("มุมมอง", views, horizontal=True)

    if view.startswith("PCA"):
        coords, evr = pca_2d(Xs, int(seed))
        pca_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "Cluster": plot_df["Cluster"]})
        fig = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster", category_orders=cluster_order, template=TEMPLATE)
        st.plotly_chart(fig)
        st.caption(f"PC1 + PC2 อธิบายความแปรปรวนของข้อมูลได้ {evr.sum() * 100:.1f}%")
    elif view == "เลือกแกนเอง 2D":
        ca, cb = st.columns(2)
        x_axis = ca.selectbox("แกน X", features, index=0)
        y_axis = cb.selectbox("แกน Y", features, index=1)
        fig = px.scatter(plot_df, x=x_axis, y=y_axis, color="Cluster", category_orders=cluster_order, template=TEMPLATE)
        st.plotly_chart(fig)
    else:
        ca, cb, cc = st.columns(3)
        x_axis = ca.selectbox("แกน X", features, index=0, key="x3")
        y_axis = cb.selectbox("แกน Y", features, index=1, key="y3")
        z_axis = cc.selectbox("แกน Z", features, index=2, key="z3")
        fig = px.scatter_3d(
            plot_df, x=x_axis, y=y_axis, z=z_axis, color="Cluster", category_orders=cluster_order, template=TEMPLATE
        )
        fig.update_traces(marker_size=3)
        st.plotly_chart(fig)

    st.subheader("สรุปลักษณะของแต่ละกลุ่ม")
    profile = cluster_profile(result, features)
    st.dataframe(profile)

    descriptions = describe_clusters(result, features)
    for c, text in descriptions.items():
        st.markdown(f"**กลุ่ม {c}** — {text}")

    left, right = st.columns(2)
    with left:
        st.subheader("สัดส่วนลูกค้าแต่ละกลุ่ม")
        sizes = result["Cluster"].value_counts().sort_index().reset_index()
        sizes.columns = ["Cluster", "Customers"]
        sizes["Cluster"] = sizes["Cluster"].astype(str)
        fig_pie = px.pie(sizes, names="Cluster", values="Customers", hole=0.45, template=TEMPLATE)
        st.plotly_chart(fig_pie)
    with right:
        st.subheader("โปรไฟล์กลุ่ม (Radar)")
        z = pd.DataFrame(Xs, columns=features).groupby(labels).mean()
        fig_radar = go.Figure()
        for c in z.index:
            vals = z.loc[c].tolist()
            fig_radar.add_trace(
                go.Scatterpolar(r=vals + vals[:1], theta=features + features[:1], fill="toself", name=f"กลุ่ม {c}")
            )
        fig_radar.update_layout(template=TEMPLATE, polar=dict(radialaxis=dict(visible=True)))
        st.plotly_chart(fig_radar)
        st.caption("ค่าบนกราฟเป็นคะแนนมาตรฐาน (Z-score) ค่ามากกว่า 0 = สูงกว่าค่าเฉลี่ยรวม")

# ---------------------------------------------------------------- Tab 3: เลือก K
with tab3:
    st.subheader("Elbow Method และ Silhouette Score")
    eval_df = cached_eval(X, int(seed))

    left, right = st.columns(2)
    with left:
        fig_elbow = px.line(eval_df, x="k", y="Inertia", markers=True, template=TEMPLATE, title="Elbow (Inertia)")
        st.plotly_chart(fig_elbow)
    with right:
        fig_sil = px.line(eval_df, x="k", y="Silhouette", markers=True, template=TEMPLATE, title="Silhouette Score")
        st.plotly_chart(fig_sil)

    best_k = int(eval_df.loc[eval_df["Silhouette"].idxmax(), "k"])
    st.success(f"ค่า K ที่ให้ Silhouette Score สูงสุดคือ K = {best_k}")
    st.dataframe(eval_df.round(3))
    st.caption("เลือก K ตรงจุดที่กราฟ Elbow หักศอก และ Silhouette สูง แล้วปรับที่แถบด้านซ้าย")

# ---------------------------------------------------------------- Tab 4: ข้อมูลผลลัพธ์
with tab4:
    st.subheader("ข้อมูลลูกค้าพร้อมหมายเลขกลุ่ม")
    st.dataframe(result)
    csv_bytes = result.to_csv(index=False).encode("utf-8-sig")
    st.download_button("ดาวน์โหลดผลลัพธ์ (CSV)", csv_bytes, file_name="customer_segments.csv", mime="text/csv")
