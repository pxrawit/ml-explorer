import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🌲 Random Forest Regressor - ทำนายยอดขายห้างสรรพสินค้า</p>', unsafe_allow_html=True)
st.markdown("โมเดล Ensemble ที่รวม Decision Tree หลายต้นเข้าด้วยกัน เพื่อทำนาย `sales_amount` ได้อย่างแม่นยำและลดโอกาส Overfitting")

# ================= Load and Prepare Data =================
@st.cache_data
def load_and_prep_data():
    try:
        df = pd.read_excel("mall_sales_eda_3000_records.xlsx")
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # ตรวจสอบว่ามีคอลัมน์ sales_amount หรือไม่
        if 'sales_amount' not in df.columns:
            st.error(f"ไม่พบคอลัมน์ 'sales_amount' ในข้อมูล")
            st.write("คอลัมน์ที่มีอยู่:", df.columns.tolist())
            st.stop()
        
        # Drop unnecessary columns
        cols_to_drop = ['record_id', 'sale_date', 'branch_a_outlier', 'special_high_sales_day']
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        df = df.drop(columns=cols_to_drop)
        
        # แปลง Boolean columns
        df['is_weekend'] = df['is_weekend'].astype(int)
        df['returned'] = df['returned'].astype(int)
        
        # One-Hot Encoding
        categorical_cols = ['day_of_week', 'branch', 'category', 'campaign', 'payment_method']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
        
        # แยก Features และ Target
        target = 'sales_amount'
        X = df.drop(columns=[target])
        y = df[target]
        
        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return df, X_train, X_test, y_train, y_test
        
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดข้อมูล: {e}")
        st.stop()

df, X_train, X_test, y_train, y_test = load_and_prep_data()

# ================= Train Random Forest =================
@st.cache_resource
def train_rf_model(X_tr, y_tr):
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    return model

model = train_rf_model(X_train, y_train)

# ================= Sidebar Input =================
st.sidebar.markdown("### 📝 กรอกข้อมูลเพื่อทำนายยอดขาย")

# Numeric inputs
customers_count = st.sidebar.number_input("จำนวนลูกค้า", min_value=0, value=100, step=10)
employee_count = st.sidebar.number_input("จำนวนพนักงาน", min_value=0, value=15, step=1)
units_sold = st.sidebar.number_input("จำนวนหน่วยที่ขายได้", min_value=0, value=50, step=5)
avg_price_per_unit = st.sidebar.number_input("ราคาเฉลี่ยต่อหน่วย", min_value=0.0, value=500.0, step=10.0)
discount_rate = st.sidebar.number_input("อัตราส่วนลด", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
satisfaction_score = st.sidebar.number_input("คะแนนความพึงพอใจ", min_value=1.0, max_value=5.0, value=4.0, step=0.1)

# Categorical inputs
day_of_week = st.sidebar.selectbox("วันในสัปดาห์", ['จันทร์', 'อังคาร', 'พุธ', 'พฤหัสบดี', 'ศุกร์', 'เสาร์', 'อาทิตย์'])
branch = st.sidebar.selectbox("สาขา", ['A', 'B', 'C', 'D', 'E'])
category = st.sidebar.selectbox("หมวดหมู่สินค้า", ['Electronics', 'Fashion', 'Food & Beverage', 'Home & Living', 'Beauty', 'Sports'])
campaign = st.sidebar.selectbox("แคมเปญ", ['None', 'Weekend Boost', 'Member Day', 'Payday Promo', 'Clearance Sale', 'Mega Sale'])
payment_method = st.sidebar.selectbox("วิธีการชำระเงิน", ['Credit Card', 'Cash', 'QR Payment', 'Mobile Banking', 'E-Wallet'])
is_weekend = st.sidebar.selectbox("เป็นวันหยุดสุดสัปดาห์", [False, True])
returned = st.sidebar.selectbox("มีการคืนสินค้า", [False, True])

# ================= Predict Button =================
if st.button("🔮 ทำนายยอดขาย", type="primary"):
    # สร้าง DataFrame จาก input
    input_data = {
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
    
    input_df = pd.DataFrame(input_data)
    
    # One-Hot Encoding
    input_df = pd.get_dummies(input_df, columns=['day_of_week', 'branch', 'category', 'campaign', 'payment_method'])
    
    # Reindex ให้ตรงกับ X_train
    input_df = input_df.reindex(columns=X_train.columns, fill_value=0)
    
    # Predict
    prediction = model.predict(input_df)[0]
    
    st.success(f"### 💰 ยอดขายที่คาดการณ์ไว้: **{prediction:,.2f} บาท**")
    
    # ================= Model Performance =================
    st.markdown("### 📊 ประสิทธิภาพของโมเดล")
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{r2:.4f}")
    col2.metric("RMSE", f"{rmse:,.2f}")
    col3.metric("MAE", f"{mae:,.2f}")
    
    # ================= Actual vs Predicted =================
    st.markdown("### 📊 เปรียบเทียบยอดขายจริง vs ที่ทำนาย")
    
    sample_idx = np.random.choice(len(X_test), size=min(200, len(X_test)), replace=False)
    
    fig_scatter = px.scatter(
        x=y_test.iloc[sample_idx],
        y=y_pred[sample_idx],
        labels={'x': 'ยอดขายจริง', 'y': 'ยอดขายที่ทำนาย'},
        opacity=0.7
    )
    fig_scatter.update_layout(height=400)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # ================= Feature Importance =================
    st.markdown("### 🏆 ความสำคัญของปัจจัย (Feature Importance)")
    
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
        color_continuous_scale='Viridis'
    )
    fig_imp.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_imp, use_container_width=True)