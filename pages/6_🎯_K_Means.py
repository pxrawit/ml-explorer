import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import io

st.set_page_config(page_title="K-Means Clustering", page_icon="🎯", layout="wide")

# ================= Custom CSS =================
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #e67e22;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: rgba(230, 126, 34, 0.1);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #e67e22;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🎯 K-Means Clustering</p>', unsafe_allow_html=True)
st.markdown("อัปโหลดไฟล์ CSV ของคุณเพื่อจัดกลุ่มข้อมูลด้วยอัลกอริทึม K-Means")

# ================= Sidebar: Upload & Parameters =================
st.sidebar.header("📁 อัปโหลดข้อมูล")
uploaded_file = st.sidebar.file_uploader("เลือกไฟล์ CSV", type=["csv"])

# ถ้าไม่ได้อัปโหลด ใช้ข้อมูลตัวอย่าง
if uploaded_file is None:
    st.info("💡 **ยังไม่มีไฟล์ CSV** กำลังแสดงข้อมูลตัวอย่างจาก Mall Sales (สามารถอัปโหลดไฟล์ของคุณได้ทางซ้ายมือ)")
    try:
        df_raw = pd.read_excel("mall_sales_eda_3000_records.xlsx")
        # เลือกเฉพาะคอลัมน์ตัวเลข
        df_raw = df_raw.select_dtypes(include=[np.number])
        # ลบคอลัมน์ที่ไม่ใช่ features
        cols_to_drop = ['record_id', 'sales_amount', 'cost_amount', 'gross_profit', 
                        'branch_a_outlier', 'special_high_sales_day']
        df_raw = df_raw.drop(columns=[c for c in cols_to_drop if c in df_raw.columns], errors='ignore')
        df_raw = df_raw.dropna()
        source_name = "Mall Sales (ตัวอย่าง)"
    except Exception as e:
        st.error(f"❌ ไม่พบไฟล์ตัวอย่าง: {e}")
        st.stop()
else:
    try:
        df_raw = pd.read_csv(uploaded_file)
        source_name = uploaded_file.name
    except Exception as e:
        st.error(f"❌ ไม่สามารถอ่านไฟล์ได้: {e}")
        st.stop()

# ================= เลือก Features (เฉพาะ Numeric) =================
numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) < 2:
    st.error("❌ ไฟล์ต้องมีคอลัมน์ตัวเลขอย่างน้อย 2 คอลัมน์เพื่อทำ K-Means")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("⚙️ ตั้งค่า K-Means")

st.sidebar.markdown(f"**📊 แหล่งข้อมูล:** {source_name}")
st.sidebar.markdown(f"**🔢 คอลัมน์ตัวเลขที่พบ:** {len(numeric_cols)}")

# เลือก K
k_value = st.sidebar.slider("จำนวนคลัสเตอร์ (K)", min_value=2, max_value=15, value=3)

# เลือก features สำหรับ clustering
selected_features = st.sidebar.multiselect(
    "เลือก Features สำหรับ Clustering",
    options=numeric_cols,
    default=numeric_cols[:min(5, len(numeric_cols))],
    help="เลือกคอลัมน์ตัวเลขที่ต้องการใช้จัดกลุ่ม"
)

if len(selected_features) < 2:
    st.warning("⚠️ กรุณาเลือกอย่างน้อย 2 features")
    st.stop()

# เลือก 2 features สำหรับ visualization (ป้องกัน duplicate)
st.sidebar.markdown("---")
st.sidebar.markdown("**📊 Visualization (2D)**")

col1_viz = st.sidebar.selectbox("แกน X", selected_features, index=0)

# สร้างรายการสำหรับแกน Y โดยตัด col1_viz ออก
available_features_for_y = [f for f in selected_features if f != col1_viz]

if len(available_features_for_y) == 0:
    st.sidebar.error("❌ ต้องมีอย่างน้อย 2 features ที่แตกต่างกัน")
    st.stop()

col2_viz = st.sidebar.selectbox("แกน Y", available_features_for_y, index=0)

# ================= Preprocessing =================
df_clean = df_clean = df_raw[selected_features].dropna()

if len(df_clean) == 0:
    st.error("❌ ไม่มีข้อมูลที่ใช้งานได้ (อาจมีค่า NaN ทั้งหมด)")
    st.stop()

# Scale ข้อมูล
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)

# ================= Train K-Means =================
@st.cache_resource
def train_kmeans(X, k):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    return model, labels

model, labels = train_kmeans(X_scaled, k_value)
sil_score = silhouette_score(X_scaled, labels)

# ================= แสดงผลหลัก =================
st.markdown("### 📊 ผลลัพธ์การ Clustering")

# Metrics
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("จำนวนคลัสเตอร์ (K)", k_value)
    st.markdown('</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("จำนวนข้อมูล", f"{len(df_clean):,}")
    st.markdown('</div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Inertia (WCSS)", f"{model.inertia_:,.2f}", help="ผลรวมระยะทางของจุดถึง centroid ของคลัสเตอร์นั้น")
    st.markdown('</div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Silhouette Score", f"{sil_score:.4f}", help="ยิ่งใกล้ 1 ยิ่งดี (-1 ถึง 1)")
    st.markdown('</div>', unsafe_allow_html=True)

# ================= 2D Visualization =================
st.markdown("### 🎨 Visualization (2D)")

# ตรวจสอบว่าเลือก feature ต่างกัน (ป้องกัน duplicate error)
if col1_viz == col2_viz:
    st.warning("⚠️ กรุณาเลือก Feature สำหรับแกน X และ Y ที่แตกต่างกัน")
    st.stop()

# สร้าง DataFrame สำหรับ plotting (rename เพื่อป้องกัน duplicate)
df_plot = df_clean[[col1_viz, col2_viz]].copy()
df_plot.columns = [col1_viz, col2_viz]  # ensure unique column names
df_plot['Cluster'] = 'Cluster ' + labels.astype(str)

fig = px.scatter(
    df_plot, 
    x=col1_viz, 
    y=col2_viz, 
    color='Cluster',
    title=f'K-Means Clustering (K={k_value})',
    color_discrete_sequence=px.colors.qualitative.Plotly
)

# เพิ่ม centroids
centroids_scaled = model.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)
df_centroids = pd.DataFrame(
    centroids_original, 
    columns=selected_features
)

fig.add_trace(go.Scatter(
    x=df_centroids[col1_viz],
    y=df_centroids[col2_viz],
    mode='markers',
    marker=dict(size=20, color='black', symbol='x', line=dict(width=2, color='white')),
    name='Centroids',
    hovertext=[f'Centroid {i}' for i in range(k_value)]
))

fig.update_layout(height=550)
st.plotly_chart(fig, use_container_width=True)

# ================= Elbow Method & Silhouette Analysis =================
tab1, tab2 = st.tabs(["📈 Elbow Method", "📊 Silhouette Analysis"])

with tab1:
    st.markdown("#### หา K ที่เหมาะสมด้วย Elbow Method")
    st.markdown("มองหาจุดที่กราฟ 'หักงอ' (elbow) - มักเป็นค่า K ที่ดีที่สุด")
    
    inertias = []
    K_range = range(2, min(16, len(df_clean)))
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(
        x=list(K_range), 
        y=inertias,
        mode='lines+markers',
        line=dict(color='#e67e22', width=3),
        marker=dict(size=10)
    ))
    fig_elbow.add_vline(x=k_value, line_dash="dash", line_color="red",
                        annotation_text=f"K={k_value}")
    fig_elbow.update_layout(
        title='Elbow Method',
        xaxis_title='จำนวนคลัสเตอร์ (K)',
        yaxis_title='Inertia (WCSS)',
        height=400
    )
    st.plotly_chart(fig_elbow, use_container_width=True)

with tab2:
    st.markdown("#### Silhouette Score สำหรับแต่ละ K")
    st.markdown("ค่า Silhouette ที่สูงแสดงว่าคลัสเตอร์มีความชัดเจนและแยกจากกันดี")
    
    sil_scores = []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_labels = km.fit_predict(X_scaled)
        sil_scores.append(silhouette_score(X_scaled, km_labels))
    
    fig_sil = go.Figure()
    fig_sil.add_trace(go.Scatter(
        x=list(K_range),
        y=sil_scores,
        mode='lines+markers',
        line=dict(color='#27ae60', width=3),
        marker=dict(size=10)
    ))
    fig_sil.add_vline(x=k_value, line_dash="dash", line_color="red",
                      annotation_text=f"K={k_value} (Score: {sil_score:.4f})")
    fig_sil.update_layout(
        title='Silhouette Score Analysis',
        xaxis_title='จำนวนคลัสเตอร์ (K)',
        yaxis_title='Silhouette Score',
        yaxis_range=[-1, 1],
        height=400
    )
    st.plotly_chart(fig_sil, use_container_width=True)

# ================= Cluster Statistics =================
st.markdown("### 📋 สถิติของแต่ละคลัสเตอร์")

df_result = df_clean.copy()
df_result['Cluster'] = labels

cluster_stats = df_result.groupby('Cluster').agg(['mean', 'count']).round(2)
st.dataframe(cluster_stats, use_container_width=True)

# ================= Distribution per Cluster =================
st.markdown("### 📊 การกระจายตัวของแต่ละคลัสเตอร์")

selected_stat_feature = st.selectbox(
    "เลือก Feature เพื่อดูการกระจายตัว",
    selected_features,
    index=0
)

fig_dist = px.box(
    df_result,
    x='Cluster',
    y=selected_stat_feature,
    color='Cluster',
    title=f'การกระจายตัวของ {selected_stat_feature} ในแต่ละคลัสเตอร์',
    color_discrete_sequence=px.colors.qualitative.Plotly
)
fig_dist.update_layout(height=400)
st.plotly_chart(fig_dist, use_container_width=True)

# ================= Download Results =================
st.markdown("### 📥 ดาวน์โหลดผลลัพธ์")

df_download = df_raw.copy()
df_download = df_download.iloc[:len(df_result)]  # ตัดให้ตรงกับข้อมูลที่ clean แล้ว
df_download['Cluster'] = labels

csv = df_download.to_csv(index=False).encode('utf-8-sig')
st.download_button(
    label="📥 ดาวน์โหลดผลลัพธ์ (CSV พร้อม Cluster Label)",
    data=csv,
    file_name=f"kmeans_results_k{k_value}.csv",
    mime="text/csv",
    use_container_width=True
)

# ================= ข้อมูลเชิงลึก =================
st.markdown("### 💡 คำแนะนำในการตีความผล")

with st.expander("📖 วิธีอ่านผลลัพธ์"):
    st.markdown(f"""
    **🎯 Inertia (WCSS) = {model.inertia_:,.2f}**
    - ผลรวมระยะทางของทุกจุดถึง centroid ของคลัสเตอร์ตัวเอง
    - ค่ายิ่งน้อย = คลัสเตอร์ยิ่งแน่น (แต่ถ้า K มากเกินไปก็จะน้อยลงเรื่อยๆ)
    
    **📊 Silhouette Score = {sil_score:.4f}**
    - วัดว่าแต่ละจุดอยู่ใกล้คลัสเตอร์ตัวเองมากกว่าคลัสเตอร์อื่นแค่ไหน
    - ช่วงค่า: -1 ถึง 1
    - **> 0.7** = คลัสเตอร์ชัดเจนดีมาก
    - **0.5 - 0.7** = ดี
    - **0.25 - 0.5** = ปานกลาง
    - **< 0.25** = คลัสเตอร์ไม่ชัดเจน ควรปรับ K
    
    **🔍 การเลือก K ที่เหมาะสม:**
    - ใช้ **Elbow Method**: มองหาจุดที่กราฟหักงอ
    - ใช้ **Silhouette Score**: เลือก K ที่ให้ค่าสูงสุด
    - พิจารณาจาก **ความหมายทางธุรกิจ** ของคลัสเตอร์
    
    **📌 Centroids (จุดศูนย์กลางคลัสเตอร์):**
    """)
    
    # แสดง centroids ในรูป DataFrame
    centroids_df = pd.DataFrame(
        centroids_original,
        columns=selected_features,
        index=[f'Cluster {i}' for i in range(k_value)]
    ).round(2)
    st.dataframe(centroids_df, use_container_width=True)

st.markdown("---")
st.caption("🎯 K-Means Clustering | อัปโหลด CSV ของคุณเพื่อวิเคราะห์")