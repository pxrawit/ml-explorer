import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

st.set_page_config(page_title="KNN - California Housing", page_icon="📍", layout="wide")

# ========== Custom CSS ==========
st.markdown("""
<style>
    .prediction-box {
        background: linear-gradient(135deg, #00c9ff 0%, #92fe9d 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: #000;
        margin: 20px 0;
    }
    .prediction-value {
        font-size: 3em;
        font-weight: bold;
        margin: 10px 0;
    }
    .metric-card {
        background: rgba(255,255,255,0.05);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #00c9ff;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 📍 KNN - California Housing Price Prediction")
st.markdown("### ทำนายราคาบ้านในแคลิฟอร์เนียด้วย K-Nearest Neighbors Regressor")

# ========== โหลดข้อมูล ==========
@st.cache_data
def load_data():
    # ลองหลาย path
    paths = [
        "california_housing_test.csv",
        "data/california_housing_test.csv",
        "../california_housing_test.csv"
    ]
    for path in paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df
    # ถ้าไม่เจอจริงๆ ให้ใช้ sklearn
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing(as_frame=True)
    return data.frame.reset_index(drop=True)

try:
    df = load_data()
    st.success(f"✅ โหลดข้อมูลสำเร็จ: {len(df):,} แถว")
except Exception as e:
    st.error(f"❌ โหลดข้อมูลไม่สำเร็จ: {e}")
    st.stop()

# ========== แสดงข้อมูล ==========
with st.expander("📊 ดูข้อมูลตัวอย่าง", expanded=False):
    st.dataframe(df.head(10))
    st.write(f"**จำนวนข้อมูล:** {len(df):,} แถว")
    st.write(f"**จำนวน Features:** {len(df.columns) - 1}")
    
    st.markdown("### 📈 สถิติสรุป")
    st.dataframe(df.describe())

# ========== แยก Features/Target ==========
X = df.drop(columns=['median_house_value'])
y = df['median_house_value']
feature_names = X.columns.tolist()

# ========== Train/Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========== Scale ข้อมูล (สำคัญมากสำหรับ KNN!) ==========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== Sidebar: ปรับพารามิเตอร์ ==========
st.sidebar.header("⚙️ พารามิเตอร์โมเดล")
k_value = st.sidebar.slider(
    "จำนวนเพื่อนบ้าน (K)", 
    min_value=1, 
    max_value=50, 
    value=5,
    help="จำนวนเพื่อนบ้านที่ใกล้ที่สุดที่จะนำมาหาค่าเฉลี่ย"
)
weights = st.sidebar.selectbox(
    "การถ่วงน้ำหนัก",
    ["uniform", "distance"],
    help="uniform = ทุกเพื่อนบ้านมีน้ำหนักเท่ากัน, distance = ยิ่งใกล้ยิ่งมีน้ำหนักมาก"
)
metric = st.sidebar.selectbox(
    "ระยะทางแบบ",
    ["euclidean", "manhattan", "minkowski"],
    help="วิธีคำนวณระยะทางระหว่างจุด"
)

# ========== เทรนโมเดล ==========
@st.cache_resource
def train_knn(k, weights, metric, X_train, y_train):
    model = KNeighborsRegressor(n_neighbors=k, weights=weights, metric=metric)
    model.fit(X_train, y_train)
    return model

with st.spinner("กำลังเทรนโมเดล..."):
    model = train_knn(k_value, weights, metric, X_train_scaled, y_train)

# ========== ประเมินโมเดล ==========
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ========== แสดงผล Metrics ==========
st.markdown("### 📈 ผลลัพธ์การเทรนโมเดล")
col1, col2, col3, col4 = st.columns(4)
col1.metric("R² Score", f"{r2:.4f}", help="ยิ่งใกล้ 1 ยิ่งดี")
col2.metric("RMSE", f"${rmse:,.0f}", help="Root Mean Squared Error")
col3.metric("MAE", f"${mae:,.0f}", help="Mean Absolute Error")
col4.metric("MSE", f"{mse:,.0f}", help="Mean Squared Error")

st.info(f"💡 **K={k_value}** | **weights={weights}** | **metric={metric}** | **Train: {len(X_train):,} samples** | **Test: {len(X_test):,} samples**")

# ========== Visualization: Predicted vs Actual ==========
st.markdown("### 🔍 Predicted vs Actual Values")
fig_scatter = go.Figure()
fig_scatter.add_trace(go.Scatter(
    x=y_test, y=y_pred,
    mode='markers',
    marker=dict(color='#00c9ff', size=5, opacity=0.5),
    name='Predictions'
))
# เส้น y=x (ทำนายสมบูรณ์)
max_val = max(y_test.max(), y_pred.max())
fig_scatter.add_trace(go.Scatter(
    x=[0, max_val], y=[0, max_val],
    mode='lines',
    line=dict(color='red', dash='dash', width=2),
    name='Perfect Prediction'
))
fig_scatter.update_layout(
    xaxis_title="Actual Price ($)",
    yaxis_title="Predicted Price ($)",
    height=400
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ========== ส่วนทำนาย ==========
st.markdown("---")
st.markdown("## 🎯 ลองทำนายราคาบ้านของคุณเอง!")
st.markdown("ปรับค่า features ด้านล่างแล้วกดปุ่มทำนาย")

# สร้าง input สำหรับแต่ละ feature
st.sidebar.markdown("---")
st.sidebar.header("🏠 กรอกข้อมูลบ้าน")

# ค่า min/max จากข้อมูลจริง
feature_ranges = {
    'longitude': (-124.35, -114.31),
    'latitude': (32.54, 41.95),
    'housing_median_age': (1.0, 52.0),
    'total_rooms': (6.0, 39320.0),
    'total_bedrooms': (1.0, 6445.0),
    'population': (3.0, 35682.0),
    'households': (1.0, 6082.0),
    'median_income': (0.50, 15.00)
}

feature_labels = {
    'longitude': '🌍 Longitude (ลองจิจูด)',
    'latitude': '🌎 Latitude (ละติจูด)',
    'housing_median_age': '🏚️ อายุบ้านเฉลี่ย (ปี)',
    'total_rooms': '🚪 จำนวนห้องทั้งหมด',
    'total_bedrooms': '🛏️ จำนวนห้องนอน',
    'population': '👥 จำนวนประชากร',
    'households': '🏘️ จำนวนครัวเรือน',
    'median_income': '💰 รายได้เฉลี่ย (x$10,000)'
}

input_values = {}
for feature in feature_names:
    min_val, max_val = feature_ranges.get(feature, (0, 100))
    default_val = float(X[feature].median())
    
    if feature in ['longitude', 'latitude']:
        input_values[feature] = st.sidebar.slider(
            feature_labels[feature],
            min_value=float(min_val),
            max_value=float(max_val),
            value=default_val,
            step=0.01
        )
    elif feature == 'median_income':
        input_values[feature] = st.sidebar.slider(
            feature_labels[feature],
            min_value=float(min_val),
            max_value=float(max_val),
            value=default_val,
            step=0.1
        )
    else:
        input_values[feature] = st.sidebar.number_input(
            feature_labels[feature],
            min_value=float(min_val),
            max_value=float(max_val),
            value=default_val,
            step=1.0
        )

# ========== ทำนาย ==========
input_df = pd.DataFrame([input_values])
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]

# หา K เพื่อนบ้านที่ใกล้ที่สุด
distances, indices = model.kneighbors(input_scaled)
neighbors = X_train.iloc[indices[0]]
neighbor_prices = y_train.iloc[indices[0]]
neighbor_distances = distances[0]

# ========== แสดงผลทำนาย ==========
st.markdown("### 🎉 ผลทำนาย")

st.markdown(f"""
<div class="prediction-box">
    <div style="font-size:1.2em;">💵 ราคาบ้านที่ทำนายได้</div>
    <div class="prediction-value">${prediction:,.0f}</div>
    <div style="font-size:1.1em;">หรือประมาณ {prediction*36:,.0f} บาท</div>
</div>
""", unsafe_allow_html=True)

# ========== แสดงเพื่อนบ้าน K ตัว ==========
st.markdown("### 👥 เพื่อนบ้านที่ใกล้ที่สุด (K Neighbors)")
st.write(f"โมเดลดูเพื่อนบ้าน {k_value} ตัวที่ใกล้ที่สุดแล้วหาค่าเฉลี่ย:")

neighbors_df = neighbors.copy()
neighbors_df['distance'] = neighbor_distances
neighbors_df['price'] = neighbor_prices.values
neighbors_df = neighbors_df.sort_values('distance')

st.dataframe(
    neighbors_df[['longitude', 'latitude', 'median_income', 'housing_median_age', 'distance', 'price']],
    use_container_width=True
)

st.write(f"**ค่าเฉลี่ยของเพื่อนบ้าน:** ${neighbor_prices.mean():,.0f}")
st.write(f"**น้ำหนักตามระยะทาง:** {'✅ ใช้' if weights == 'distance' else '❌ ไม่ใช้'}")

# ========== Map Visualization ==========
st.markdown("### 🗺️ ตำแหน่งบนแผนที่")
fig_map = go.Figure()

# จุดข้อมูลทั้งหมด (สุ่มมา 500 จุด)
sample_idx = np.random.choice(len(X), min(500, len(X)), replace=False)
fig_map.add_trace(go.Scattergeo(
    lat=X.iloc[sample_idx]['latitude'],
    lon=X.iloc[sample_idx]['longitude'],
    marker=dict(size=4, color=y.iloc[sample_idx], colorscale='Viridis', opacity=0.5),
    name='ข้อมูลทั้งหมด',
    text=y.iloc[sample_idx].apply(lambda x: f"${x:,.0f}")
))

# จุดที่ทำนาย
fig_map.add_trace(go.Scattergeo(
    lat=[input_values['latitude']],
    lon=[input_values['longitude']],
    marker=dict(size=20, color='red', symbol='star'),
    name='🎯 จุดที่ทำนาย',
    text=[f"ทำนาย: ${prediction:,.0f}"]
))

# จุดเพื่อนบ้าน
fig_map.add_trace(go.Scattergeo(
    lat=neighbors['latitude'],
    lon=neighbors['longitude'],
    marker=dict(size=12, color='yellow', symbol='circle', line=dict(width=2, color='red')),
    name=f'👥 เพื่อนบ้าน {k_value} ตัว',
    text=[f"${p:,.0f}" for p in neighbor_prices]
))

fig_map.update_geos(
    scope='usa',
    projection_type='albers usa',
    showland=True,
    landcolor="rgb(250, 250, 250)",
    countrycolor="rgb(200, 200, 200)"
)
fig_map.update_layout(height=500, title='ตำแหน่งบ้านบนแผนที่ California')
st.plotly_chart(fig_map, use_container_width=True)

# ========== Feature Distribution ==========
st.markdown("### 📊 การกระจายตัวของ Features ที่คุณกรอก")

col_a, col_b = st.columns(2)
with col_a:
    # Income distribution
    fig_inc = px.histogram(
        X, x='median_income', nbins=50,
        title='การกระจายตัวของรายได้เฉลี่ย',
        labels={'median_income': 'Median Income (x$10k)'}
    )
    fig_inc.add_vline(x=input_values['median_income'], line_dash="dash", line_color="red",
                      annotation_text=f"คุณ: {input_values['median_income']:.2f}")
    st.plotly_chart(fig_inc, use_container_width=True)

with col_b:
    # Age distribution
    fig_age = px.histogram(
        X, x='housing_median_age', nbins=50,
        title='การกระจายตัวของอายุบ้าน',
        labels={'housing_median_age': 'Housing Median Age'}
    )
    fig_age.add_vline(x=input_values['housing_median_age'], line_dash="dash", line_color="red",
                      annotation_text=f"คุณ: {input_values['housing_median_age']:.0f}")
    st.plotly_chart(fig_age, use_container_width=True)

# ========== ข้อดี/ข้อเสีย ==========
st.markdown("---")
st.markdown("### ✅ ข้อดี / ❌ ข้อเสียของ KNN สำหรับงานนี้")

col1, col2 = st.columns(2)
with col1:
    st.success("""
    **✅ ข้อดี:**
    - เข้าใจง่าย ไม่ต้อง train แบบซับซ้อน
    - จับความสัมพันธ์แบบ non-linear ได้ดี
    - ปรับ K และ weights ได้ยืดหยุ่น
    - ทำงานได้ดีกับข้อมูลขนาดเล็ก-กลาง
    """)
with col2:
    st.error("""
    **❌ ข้อเสีย:**
    - ช้าเมื่อข้อมูลเยอะ (ต้องคำนวณระยะทางทุกจุด)
    - **ต้อง scale ข้อมูล** (สำคัญมาก!)
    - อ่อนไหวต่อ curse of dimensionality
    - ต้องเลือก K ให้เหมาะสม
    """)

st.info("""
💡 **เคล็ดลับ:**
- ลองปรับ **K** ดู: K น้อย = overfit, K มาก = underfit
- ใช้ **distance weights** มักได้ผลดีกว่า uniform
- **StandardScaler** จำเป็นมากสำหรับ KNN
- ข้อมูล California Housing มี outlier ที่ราคา 500,001 ควรกรองออกก่อนเทรนจริง
""")

st.markdown("---")
st.caption("📍 KNN California Housing Predictor | Machine Learning Explorer")