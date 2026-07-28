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
</style>
""", unsafe_allow_html=True)

st.title("📍 KNN - California Housing Price Prediction")
st.markdown("### ทำนายราคาบ้านในแคลิฟอร์เนียด้วย K-Nearest Neighbors Regressor")

@st.cache_data
def load_data():
    paths = [
        "california_housing_test.csv",
        "data/california_housing_test.csv",
        "../california_housing_test.csv"
    ]
    for path in paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing(as_frame=True)
    return data.frame.reset_index(drop=True)

try:
    df = load_data()
    st.success("✅ โหลดข้อมูลสำเร็จ: " + str(len(df)) + " แถว")
except Exception as e:
    st.error("❌ โหลดข้อมูลไม่สำเร็จ: " + str(e))
    st.stop()

with st.expander("📊 ดูข้อมูลตัวอย่าง", expanded=False):
    st.dataframe(df.head(10))
    st.write("**จำนวนข้อมูล:** " + str(len(df)) + " แถว")
    st.write("**จำนวน Features:** " + str(len(df.columns) - 1))
    st.markdown("### 📈 สถิติสรุป")
    st.dataframe(df.describe())

X = df.drop(columns=['median_house_value'])
y = df['median_house_value']
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.sidebar.header("⚙️ พารามิเตอร์โมเดล")
k_value = st.sidebar.slider("จำนวนเพื่อนบ้าน (K)", 1, 50, 5)
weights = st.sidebar.selectbox("การถ่วงน้ำหนัก", ["uniform", "distance"])
metric = st.sidebar.selectbox("ระยะทางแบบ", ["euclidean", "manhattan", "minkowski"])

@st.cache_resource
def train_knn(k, w, m, X_tr, y_tr):
    model = KNeighborsRegressor(n_neighbors=k, weights=w, metric=m)
    model.fit(X_tr, y_tr)
    return model

with st.spinner("กำลังเทรนโมเดล..."):
    model = train_knn(k_value, weights, metric, X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.markdown("### 📈 ผลลัพธ์การเทรนโมเดล")
col1, col2, col3, col4 = st.columns(4)
col1.metric("R² Score", f"{r2:.4f}")
col2.metric("RMSE", f"${rmse:,.0f}")
col3.metric("MAE", f"${mae:,.0f}")
col4.metric("MSE", f"{mse:,.0f}")

st.info(f"💡 K={k_value} | weights={weights} | metric={metric}")

st.markdown("### 🔍 Predicted vs Actual Values")
fig_scatter = go.Figure()
fig_scatter.add_trace(go.Scatter(
    x=y_test, y=y_pred, mode='markers',
    marker=dict(color='#00c9ff', size=5, opacity=0.5), name='Predictions'
))
max_val = max(float(y_test.max()), float(y_pred.max()))
fig_scatter.add_trace(go.Scatter(
    x=[0, max_val], y=[0, max_val], mode='lines',
    line=dict(color='red', dash='dash', width=2), name='Perfect Prediction'
))
fig_scatter.update_layout(xaxis_title="Actual Price ($)", yaxis_title="Predicted Price ($)", height=400)
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")
st.markdown("## 🎯 ลองทำนายราคาบ้านของคุณเอง!")

st.sidebar.markdown("---")
st.sidebar.header("🏠 กรอกข้อมูลบ้าน")

feature_ranges = {
    'longitude': (-124.35, -114.31), 'latitude': (32.54, 41.95),
    'housing_median_age': (1.0, 52.0), 'total_rooms': (6.0, 39320.0),
    'total_bedrooms': (1.0, 6445.0), 'population': (3.0, 35682.0),
    'households': (1.0, 6082.0), 'median_income': (0.50, 15.00)
}

feature_labels = {
    'longitude': '🌍 Longitude', 'latitude': '🌎 Latitude',
    'housing_median_age': '🏚️ อายุบ้านเฉลี่ย (ปี)', 'total_rooms': '🚪 จำนวนห้องทั้งหมด',
    'total_bedrooms': '🛏️ จำนวนห้องนอน', 'population': '👥 จำนวนประชากร',
    'households': '🏘️ จำนวนครัวเรือน', 'median_income': '💰 รายได้เฉลี่ย (x$10,000)'
}

input_values = {}
for feature in feature_names:
    min_val, max_val = feature_ranges.get(feature, (0, 100))
    default_val = float(X[feature].median())
    
    if feature in ['longitude', 'latitude']:
        input_values[feature] = st.sidebar.slider(feature_labels[feature], float(min_val), float(max_val), default_val, 0.01)
    elif feature == 'median_income':
        input_values[feature] = st.sidebar.slider(feature_labels[feature], float(min_val), float(max_val), default_val, 0.1)
    else:
        input_values[feature] = st.sidebar.number_input(feature_labels[feature], float(min_val), float(max_val), default_val, 1.0)

input_df = pd.DataFrame([input_values])
input_scaled = scaler.transform(input_df)
prediction = float(model.predict(input_scaled)[0])

st.markdown("### 🎉 ผลทำนาย")
pred_thb = prediction * 36
html_content = f"""
<div class="prediction-box">
    <div style="font-size:1.2em;">💵 ราคาบ้านที่ทำนายได้</div>
    <div class="prediction-value">${prediction:,.0f}</div>
    <div style="font-size:1.1em;">หรือประมาณ {pred_thb:,.0f} บาท</div>
</div>
"""
st.markdown(html_content, unsafe_allow_html=True)

st.markdown("### 📊 การกระจายตัวของ Features ที่คุณกรอก")
col_a, col_b = st.columns(2)
with col_a:
    fig_inc = px.histogram(X, x='median_income', nbins=50, title='การกระจายตัวของรายได้เฉลี่ย')
    fig_inc.add_vline(x=input_values['median_income'], line_dash="dash", line_color="red")
    st.plotly_chart(fig_inc, use_container_width=True)

with col_b:
    fig_age = px.histogram(X, x='housing_median_age', nbins=50, title='การกระจายตัวของอายุบ้าน')
    fig_age.add_vline(x=input_values['housing_median_age'], line_dash="dash", line_color="red")
    st.plotly_chart(fig_age, use_container_width=True)

st.markdown("---")
st.caption("📍 KNN California Housing Predictor | Machine Learning Explorer")