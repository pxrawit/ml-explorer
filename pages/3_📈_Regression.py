import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
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

st.markdown('<p class="main-title">📈 Regression: ทำนายยอดขาย (Sales Amount)</p>', unsafe_allow_html=True)
st.markdown("ใช้โมเดล Machine Learning เพื่อทำนายยอดขายจากปัจจัยต่างๆ ในห้างสรรพสินค้า")

# ================= Load Data =================
@st.cache_data
def load_data():
    paths = ["mall_sales_eda_3000_records.xlsx", "data/mall_sales_eda_3000_records.xlsx"]
    for path in paths:
        if os.path.exists(path):
            return pd.read_excel(path)
    st.error("❌ ไม่พบไฟล์ mall_sales_eda_3000_records.xlsx")
    st.stop()

df = load_data()

# ================= Data Preprocessing =================
# ลบคอลัมน์ที่ไม่จำเป็นออก
cols_to_drop = ['record_id', 'sale_date', 'branch_a_outlier', 'special_high_sales_day']
df_clean = df.drop(columns=[col for col in cols_to_drop if col in df.columns]).dropna()

# แปลง TRUE/FALSE เป็น 1/0
df_clean['is_weekend'] = df_clean['is_weekend'].map({'TRUE': 1, 'FALSE': 0})
df_clean['returned'] = df_clean['returned'].map({'TRUE': 1, 'FALSE': 0})

# Target variable
target = 'sales_amount'

# One-Hot Encoding สำหรับ Categorical variables
categorical_cols = ['branch', 'category', 'campaign', 'day_of_week', 'payment_method']
df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)

# แยก X และ y
X = df_encoded.drop(columns=[target])
y = df_encoded[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================= Sidebar: User Input & Model Selection =================
st.sidebar.header("⚙️ ตั้งค่าโมเดลและข้อมูล")

model_choice = st.sidebar.selectbox(
    "เลือกอัลกอริทึม",
    ["Linear Regression", "Random Forest Regressor"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📝 กรอกข้อมูลเพื่อทำนาย")

# สร้าง Input Widgets ตามคอลัมน์ใน X (หลังทำ One-Hot Encoding)
user_inputs = {}
for col in X.columns:
    if any(cat in col for cat in ['branch_', 'category_', 'campaign_', 'day_of_week_', 'payment_method_']):
        # เป็นคอลัมน์ที่ผ่านการ One-Hot Encoding แล้ว (เช่น branch_B, category_Electronics)
        # เราจะสร้าง Checkbox แทน
        original_col = "_".join(col.split("_")[1:]) # ดึงชื่อเดิมกลับมา (เช่น branch_B -> B)
        # จัดกลุ่มให้สวยงาม (เอาเฉพาะ unique แรกๆ ของแต่ละประเภทมาแสดง หรือใช้ checkbox ธรรมดา)
        # เพื่อความง่ายและ minimal เราจะใช้ Checkbox โดยตรง
        user_inputs[col] = st.sidebar.checkbox(f"เป็น {col.replace('_', ' ')}", value=False)
    else:
        # เป็นคอลัมน์ตัวเลข
        min_val, max_val = float(X[col].min()), float(X[col].max())
        user_inputs[col] = st.sidebar.number_input(
            col.replace('_', ' ').title(), 
            min_value=min_val, 
            max_value=max_val, 
            value=float(X[col].median()),
            step=1.0 if col in ['customers_count', 'employee_count', 'units_sold'] else 0.01
        )

# ================= Model Training =================
@st.cache_resource
def train_model(model_name, X_tr, y_tr):
    if model_name == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_tr, y_tr)
    return model

model = train_model(model_choice, X_train, y_train)

# ประเมินโมเดล
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# ================= Main Content: Prediction & Metrics =================
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("R² Score", f"{r2:.4f}", help="ยิ่งใกล้ 1 ยิ่งทำนายได้แม่นยำ")
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("MAE", f"{mae:,.0f}", help="ค่าความคลาดเคลื่อนเฉลี่ย")
    st.markdown('</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("RMSE", f"{rmse:,.0f}", help="Root Mean Squared Error")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ================= Prediction Button & Result =================
if st.button("🔮 ทำนายยอดขาย (Predict)", type="primary", use_container_width=True):
    # สร้าง DataFrame จาก input ผู้ใช้
    input_df = pd.DataFrame([user_inputs])
    
    # ทำความแน่ใจว่าคอลัมน์ตรงกับตอนเทรน (กรณี missing columns จาก get_dummies)
    input_df = input_df.reindex(columns=X.columns, fill_value=0)
    
    # ทำนาย
    predicted_sales = model.predict(input_df)[0]
    
    st.success(f"### 💰 ยอดขายที่คาดการณ์ไว้: **{predicted_sales:,.2f} บาท**")
    
    # แสดง Actual vs Predicted Scatter Plot (สุ่มมา 200 จุดเพื่อไม่ให้หนัก)
    st.markdown("### 📊 Actual vs Predicted (สุ่มตัวอย่าง 200 รายการ)")
    sample_idx = np.random.choice(len(X_test), size=200, replace=False)
    
    fig = px.scatter(
        x=y_test.iloc[sample_idx], 
        y=y_pred[sample_idx],
        labels={'x': 'ยอดขายจริง (Actual)', 'y': 'ยอดขายที่ทำนาย (Predicted)'},
        opacity=0.7,
        trendline="ols",
        trendline_color_override="red"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ================= Feature Importance (ถ้าเป็น Random Forest) =================
if model_choice == "Random Forest Regressor":
    st.markdown("---")
    st.markdown("### 🏆 ความสำคัญของปัจจัย (Feature Importance)")
    
    importances = model.feature_importances_
    feature_names = X.columns
    
    # ดึง Top 10 features
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values('Importance', ascending=False).head(10)
    
    # ทำความสะอาดชื่อ feature ให้สวยงาม (ลบ prefix ที่เกิดจาก get_dummies)
    importance_df['Feature'] = importance_df['Feature'].apply(lambda x: x.replace('branch_', 'สาขา ').replace('category_', 'หมวดหมู่ ').replace('campaign_', 'แคมเปญ ').replace('day_of_week_', 'วัน ').replace('payment_method_', 'การชำระเงิน '))
    
    fig_imp = px.bar(
        importance_df, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        color='Importance',
        color_continuous_scale='Blues'
    )
    fig_imp.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_imp, use_container_width=True)