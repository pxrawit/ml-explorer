import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

st.set_page_config(page_title="Random Forest - Mall Sales", page_icon="🌲", layout="wide")

# ================= Custom CSS =================
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: rgba(46, 204, 113, 0.1);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #2ecc71;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🌲 Random Forest Regressor - ทำนายยอดขายห้างสรรพสินค้า</p>', unsafe_allow_html=True)
st.markdown("โมเดล Ensemble ที่รวม Decision Tree หลายต้นเข้าด้วยกัน เพื่อทำนาย `sales_amount` ได้อย่างแม่นยำและลดโอกาส Overfitting")

# ================= 1. โหลดและเตรียมข้อมูล =================
@st.cache_data
def load_and_prep_data():
    paths = ["mall_sales_eda_3000_records.xlsx", "data/mall_sales_eda_3000_records.xlsx"]
    for path in paths:
        if os.path.exists(path):
            df = pd.read_excel(path)
            break
    else:
        st.error("ไม่พบไฟล์ mall_sales_eda_3000_records.xlsx")
        st.stop()
    
    # ตัดคอลัมน์ที่ไม่จำเป็นออก (ID, Date, Target Leakage, EDA columns)
    cols_to_drop = ['record_id', 'sale_date', 'sales_amount', 'cost_amount', 
                    'gross_profit', 'branch_a_outlier', 'special_high_sales_day']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # แปลง Boolean columns เป็น int
    bool_cols = ['is_weekend', 'returned']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0}).fillna(0).astype(int)
    
    # One-Hot Encoding สำหรับ Categorical variables
    categorical_cols = ['day_of_week', 'branch', 'category', 'campaign', 'payment_method']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    
    # บังคับให้ทุกคอลัมน์เป็น numeric
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # กำหนด Target และ Features
    target = 'sales_amount'
    X = df.drop(columns=[target])
    y = df[target]
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return df, X_train, X_test, y_train, y_test

df, X_train, X_test, y_train, y_test = load_and_prep_data()

# ================= 2. เทรนโมเดล Random Forest =================
@st.cache_resource
def train_rf_model(X_tr, y_tr):
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    return model

model = train_rf_model(X_train, y_train)

# ================= 3. Sidebar: กรอกข้อมูลสำหรับทำนาย =================
st.sidebar.markdown("### 📝 กรอกข้อมูลเพื่อทำนายยอดขาย")
st.sidebar.markdown("---")

# ข้อมูลเชิงตัวเลข
st.sidebar.markdown("#### 🔢 ข้อมูลเชิงปริมาณ")
customers_count = st.sidebar.number_input("จำนวนลูกค้า (customers_count)", min_value=0, value=100, step=10)
employee_count = st.sidebar.number_input("จำนวนพนักงาน (employee_count)", min_value=0, value=15, step=1)
units_sold = st.sidebar.number_input("จำนวนหน่วยที่ขายได้ (units_sold)", min_value=0, value=50, step=5)
avg_price_per_unit = st.sidebar.number_input("ราคาเฉลี่ยต่อหน่วย (avg_price_per_unit)", min_value=0.0, value=500.0, step=10.0)
discount_rate = st.sidebar.number_input("อัตราส่วนลด (discount_rate)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
satisfaction_score = st.sidebar.number_input("คะแนนความพึงพอใจ (satisfaction_score)", min_value=1.0, max_value=5.0, value=4.0, step=0.1)

# ข้อมูลเชิงหมวดหมู่
st.sidebar.markdown("#### 🏷️ ข้อมูลเชิงหมวดหมู่")
day_of_week = st.sidebar.selectbox("วันในสัปดาห์ (day_of_week)", 
                                    ['จันทร์', 'อังคาร', 'พุธ', 'พฤหัสบดี', 'ศุกร์', 'เสาร์', 'อาทิตย์'])
branch = st.sidebar.selectbox("สาขา (branch)", ['A', 'B', 'C', 'D', 'E'])
category = st.sidebar.selectbox("หมวดหมู่สินค้า (category)", 
                                 ['Electronics', 'Fashion', 'Food & Beverage', 'Home & Living', 'Beauty', 'Sports'])
campaign = st.sidebar.selectbox("แคมเปญ (campaign)", 
                                 ['None', 'Weekend Boost', 'Member Day', 'Payday Promo', 'Clearance Sale', 'Mega Sale'])
payment_method = st.sidebar.selectbox("วิธีการชำระเงิน (payment_method)", 
                                       ['Credit Card', 'Cash', 'QR Payment', 'Mobile Banking', 'E-Wallet'])
is_weekend = st.sidebar.selectbox("เป็นวันหยุดสุดสัปดาห์ (is_weekend)", [False, True])
returned = st.sidebar.selectbox("มีการคืนสินค้า (returned)", [False, True])

# ================= 4. ปุ่มทำนายผล =================
st.markdown("---")
if st.button("🔮 ทำนายยอดขาย (Predict Sales Amount)", type="primary", use_container_width=True):
    
    # สร้าง DataFrame จากข้อมูลที่ผู้ใช้กรอก
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
    
    # One-Hot Encoding ข้อมูล input
    input_df = pd.get_dummies(input_df, columns=['day_of_week', 'branch', 'category', 'campaign', 'payment_method'])
    
    # บังคับเป็น numeric
    input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # จัดเรียงคอลัมน์ให้ตรงกับ X_train และเติม 0 สำหรับคอลัมน์ที่หายไป
    input_df = input_df.reindex(columns=X_train.columns, fill_value=0)
    
    # ทำนายผล
    prediction = model.predict(input_df)[0]
    
    # แสดงผลลัพธ์
    st.success(f"### 💰 ยอดขายที่คาดการณ์ไว้: **{prediction:,.2f} บาท**")
    
    st.markdown("---")
    
    # ================= 5. แสดง Feature Importance =================
    st.markdown("### 🏆 ความสำคัญของปัจจัย (Feature Importance)")
    st.markdown("ปัจจัยใดที่มีผลต่อยอดขายมากที่สุดตามมุมมองของ Random Forest")
    
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(10)
    
    fig_imp = px.bar(
        importance_df, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        color='Importance',
        color_continuous_scale='Viridis',
        text='Importance'
    )
    fig_imp.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig_imp.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_imp, use_container_width=True)

    # ================= 6. Actual vs Predicted =================
    st.markdown("### 📊 เปรียบเทียบยอดขายจริง vs ที่ทำนาย (สุ่มตัวอย่าง 200 รายการ)")
    
    sample_size = min(200, len(X_test))
    sample_idx = np.random.choice(len(X_test), size=sample_size, replace=False)
    
    fig_scatter = px.scatter(
        x=y_test.iloc[sample_idx],
        y=model.predict(X_test.iloc[sample_idx]),
        labels={'x': 'ยอดขายจริง (Actual)', 'y': 'ยอดขายที่ทำนาย (Predicted)'},
        opacity=0.7
    )
    fig_scatter.update_layout(height=400)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ================= 7. แสดง Metric สรุป =================
    st.markdown("### 📈 สรุปประสิทธิภาพของโมเดล")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("R² Score", f"{r2_score(y_test, model.predict(X_test)):.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, model.predict(X_test))):,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("MAE", f"{mean_absolute_error(y_test, model.predict(X_test)):,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)