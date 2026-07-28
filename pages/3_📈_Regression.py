import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

st.set_page_config(page_title="Regression - Mall Sales", page_icon="📈", layout="wide")

# ================= Custom CSS =================
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: rgba(31, 119, 180, 0.1);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">📈 Linear Regression - ทำนายยอดขายห้างสรรพสินค้า</p>', unsafe_allow_html=True)
st.markdown("ใช้สมการเชิงเส้นเพื่อหาความสัมพันธ์ระหว่างปัจจัยและยอดขาย (sales_amount)")

# ================= Load Data =================
@st.cache_data
def load_data():
    paths = ["mall_sales_eda_3000_records.xlsx", "data/mall_sales_eda_3000_records.xlsx"]
    for path in paths:
        if os.path.exists(path):
            return pd.read_excel(path)
    st.error("ไม่พบไฟล์ mall_sales_eda_3000_records.xlsx")
    st.stop()

df = load_data()

# ================= Preprocessing =================
df['is_weekend'] = df['is_weekend'].astype(int)
df['returned'] = df['returned'].astype(int)

# One-Hot Encoding
categorical_cols = ['day_of_week', 'branch', 'category', 'campaign', 'payment_method']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

# Target & Features
target = 'sales_amount'
X = df_encoded.drop(columns=[target])
y = df_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================= Train Linear Regression =================
@st.cache_resource
def train_lr(X_tr, y_tr):
    model = LinearRegression()
    model.fit(X_tr, y_tr)
    return model

model = train_lr(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# ================= Sidebar Input =================
st.sidebar.markdown("### 📝 กรอกข้อมูลเพื่อทำนายยอดขาย")

st.sidebar.markdown("#### 🔢 ข้อมูลเชิงปริมาณ")
customers_count = st.sidebar.number_input("จำนวนลูกค้า (customers_count)", min_value=0, value=100, step=10)
employee_count = st.sidebar.number_input("จำนวนพนักงาน (employee_count)", min_value=0, value=15, step=1)
units_sold = st.sidebar.number_input("จำนวนหน่วยที่ขายได้ (units_sold)", min_value=0, value=50, step=5)
avg_price_per_unit = st.sidebar.number_input("ราคาเฉลี่ยต่อหน่วย (avg_price_per_unit)", min_value=0.0, value=500.0, step=10.0)
discount_rate = st.sidebar.number_input("อัตราส่วนลด (discount_rate)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
satisfaction_score = st.sidebar.number_input("คะแนนความพึงพอใจ (satisfaction_score)", min_value=1.0, max_value=5.0, value=4.0, step=0.1)

st.sidebar.markdown("#### 🏷️ ข้อมูลเชิงหมวดหมู่")
day_of_week = st.sidebar.selectbox("วันในสัปดาห์", df['day_of_week'].unique())
branch = st.sidebar.selectbox("สาขา", df['branch'].unique())
category = st.sidebar.selectbox("หมวดหมู่สินค้า", df['category'].unique())
campaign = st.sidebar.selectbox("แคมเปญ", df['campaign'].unique())
payment_method = st.sidebar.selectbox("วิธีการชำระเงิน", df['payment_method'].unique())
is_weekend = st.sidebar.selectbox("เป็นวันหยุดสุดสัปดาห์", [True, False])
returned = st.sidebar.selectbox("มีการคืนสินค้า", [False, True])

# ================= Predict =================
if st.button("🔮 ทำนายยอดขาย (Predict Sales Amount)", type="primary", use_container_width=True):
    # Build input
    user_data = {
        'customers_count': [customers_count],
        'employee_count': [employee_count],
        'units_sold': [units_sold],
        'avg_price_per_unit': [avg_price_per_unit],
        'discount_rate': [discount_rate],
        'satisfaction_score': [satisfaction_score],
        'day_of_week': [day_of_week],
        'branch': [branch],
        'category': [category],
        'campaign': [campaign],
        'payment_method': [payment_method],
        'is_weekend': [int(is_weekend)],
        'returned': [int(returned)]
    }
    input_df = pd.DataFrame(user_data)
    input_encoded = pd.get_dummies(input_df, columns=['day_of_week', 'branch', 'category', 'campaign', 'payment_method'])
    input_encoded = input_encoded.reindex(columns=X_train.columns, fill_value=0)
    
    prediction = model.predict(input_encoded)[0]
    
    st.success(f"### 💰 ยอดขายที่คาดการณ์ไว้: **{prediction:,.2f} บาท**")
    
    # ================= Metrics =================
    st.markdown("### 📈 สรุปประสิทธิภาพของโมเดล")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("R² Score", f"{r2:.4f}", help="ยิ่งใกล้ 1 ยิ่งอธิบายความแปรปรวนได้ดี")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("RMSE", f"{rmse:,.2f}", help="ค่าคลาดเคลื่อนกำลังสองเฉลี่ยราก")
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("MAE", f"{mae:,.2f}", help="ค่าคลาดเคลื่อนสัมบูรณ์เฉลี่ย")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ================= Actual vs Predicted =================
    st.markdown("### 📊 เปรียบเทียบยอดขายจริง vs ที่ทำนาย (สุ่มตัวอย่าง 200 รายการ)")
    sample_idx = np.random.choice(len(X_test), size=min(200, len(X_test)), replace=False)
    
    fig_scatter = px.scatter(
        x=y_test.iloc[sample_idx],
        y=y_pred[sample_idx],
        labels={'x': 'ยอดขายจริง (Actual)', 'y': 'ยอดขายที่ทำนาย (Predicted)'},
        opacity=0.7,
        trendline="ols",
        trendline_color_override="red"
    )
    fig_scatter.update_layout(height=400)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # ================= Coefficients =================
    st.markdown("### 🔍 สัมประสิทธิ์ของโมเดล (Coefficients)")
    st.markdown("แสดงปัจจัยที่มีผลต่อยอดขาย (ค่าสัมประสิทธิ์จากสมการเชิงเส้น)")
    
    coef_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False).head(15)
    
    fig_coef = px.bar(
        coef_df,
        x='Coefficient',
        y='Feature',
        orientation='h',
        color='Coefficient',
        color_continuous_scale='RdBu',
        color_continuous_midpoint=0
    )
    fig_coef.update_layout(height=500)
    st.plotly_chart(fig_coef, use_container_width=True)