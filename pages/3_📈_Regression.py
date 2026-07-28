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

@st.cache_data
def load_data():
    paths = ["mall_sales_eda_3000_records.xlsx", "data/mall_sales_eda_3000_records.xlsx", "mall_sales_eda_3000_records.csv"]
    for path in paths:
        if os.path.exists(path):
            if path.endswith('.csv'):
                return pd.read_csv(path)
            else:
                return pd.read_excel(path)
    st.error("ไม่พบไฟล์ mall_sales_eda_3000_records")
    st.stop()

df = load_data()

# 1. กำหนด Target
target = 'sales_amount'
y = df[target]

# 2. ทิ้งคอลัมน์ที่ไม่ได้ใช้ทำนาย (รวมถึง cost และ gross_profit เพื่อป้องกัน Data Leakage)
cols_to_drop = ['record_id', 'sale_date', 'sales_amount', 'cost_amount', 'gross_profit']
X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# 3. แปลง Boolean เป็น Integer (จัดการกรณีที่เป็น string 'TRUE'/'FALSE')
bool_cols = ['is_weekend', 'returned', 'branch_a_outlier', 'special_high_sales_day']
for col in bool_cols:
    if col in X.columns:
        X[col] = X[col].astype(str).map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0}).fillna(0).astype(int)

# 4. One-Hot Encoding สำหรับ Categorical
categorical_cols = ['day_of_week', 'branch', 'category', 'campaign', 'payment_method']
X = pd.get_dummies(X, columns=categorical_cols, drop_first=False, dtype=int)

# 5. บังคับให้ทุกคอลัมน์เป็น numeric (ป้องกัน error จาก string ที่หลงเหลือ)
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Sidebar for Prediction
st.sidebar.markdown("### 📝 กรอกข้อมูลเพื่อทำนายยอดขาย")

st.sidebar.markdown("#### 🔢 ข้อมูลเชิงปริมาณ")
customers_count = st.sidebar.number_input("จำนวนลูกค้า", min_value=0, value=int(X['customers_count'].median()), step=10)
employee_count = st.sidebar.number_input("จำนวนพนักงาน", min_value=0, value=int(X['employee_count'].median()), step=1)
units_sold = st.sidebar.number_input("จำนวนหน่วยที่ขายได้", min_value=0, value=int(X['units_sold'].median()), step=5)
avg_price_per_unit = st.sidebar.number_input("ราคาเฉลี่ยต่อหน่วย", min_value=0.0, value=float(X['avg_price_per_unit'].median()), step=10.0)
discount_rate = st.sidebar.number_input("อัตราส่วนลด", min_value=0.0, max_value=1.0, value=float(X['discount_rate'].median()), step=0.01)
satisfaction_score = st.sidebar.number_input("คะแนนความพึงพอใจ", min_value=1.0, max_value=5.0, value=float(X['satisfaction_score'].median()), step=0.1)

st.sidebar.markdown("#### 🏷️ ข้อมูลเชิงหมวดหมู่")
day_of_week = st.sidebar.selectbox("วันในสัปดาห์", df['day_of_week'].unique())
branch = st.sidebar.selectbox("สาขา", df['branch'].unique())
category = st.sidebar.selectbox("หมวดหมู่สินค้า", df['category'].unique())
campaign = st.sidebar.selectbox("แคมเปญ", df['campaign'].unique())
payment_method = st.sidebar.selectbox("วิธีการชำระเงิน", df['payment_method'].unique())
is_weekend = st.sidebar.selectbox("เป็นวันหยุดสุดสัปดาห์", [True, False])
returned = st.sidebar.selectbox("มีการคืนสินค้า", [False, True])
branch_a_outlier = st.sidebar.selectbox("เป็นสาขา A Outlier", [False, True])
special_high_sales_day = st.sidebar.selectbox("เป็นวันยอดขายสูงพิเศษ", [False, True])

if st.button("🔮 ทำนายยอดขาย (Predict)", type="primary", use_container_width=True):
    # สร้าง DataFrame จากข้อมูลที่ผู้ใช้กรอก (ใช้ชื่อคอลัมน์เดิม)
    input_df = pd.DataFrame([{
        'customers_count': customers_count,
        'employee_count': employee_count,
        'units_sold': units_sold,
        'avg_price_per_unit': avg_price_per_unit,
        'discount_rate': discount_rate,
        'satisfaction_score': satisfaction_score,
        'is_weekend': is_weekend,
        'returned': returned,
        'branch_a_outlier': branch_a_outlier,
        'special_high_sales_day': special_high_sales_day,
        'day_of_week': day_of_week,
        'branch': branch,
        'category': category,
        'campaign': campaign,
        'payment_method': payment_method
    }])
    
    # ใช้การแปลงข้อมูลแบบเดียวกับตอนเทรน
    for col in bool_cols:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str).map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0}).fillna(0).astype(int)
            
    input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=False, dtype=int)
    
    # ให้คอลัมน์ตรงกับ X_train ทุกประการ (เติม 0 ถ้าขาด, เรียงลำดับให้ตรง)
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
    input_df = input_df[X.columns]
    input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    prediction = model.predict(input_df)[0]
    
    st.success(f"### 💰 ยอดขายที่คาดการณ์ไว้: **{prediction:,.2f}**")
    
    # Metrics
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
    
    # Actual vs Predicted
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