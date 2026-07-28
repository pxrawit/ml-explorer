import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing

# ตั้งค่าหน้าเว็บให้เรียบง่าย
st.set_page_config(page_title="KNN Housing", page_icon="🏠", layout="centered")

# ========== 1. โหลดและเตรียมข้อมูล ==========
@st.cache_data
def load_data():
    # ใช้ข้อมูลในตัว Scikit-Learn เพื่อความชัวร์ (ไม่ต้องพึ่งไฟล์ CSV)
    data = fetch_california_housing(as_frame=True)
    return data.frame

df = load_data()

# แยกข้อมูล (ชื่อ Feature ตามมาตรฐานของ Scikit-Learn)
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal'] # ราคาบ้าน (หน่วย: $100,000)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== 2. ส่วนหัวและพารามิเตอร์ ==========
st.title("🏠 ประเมินราคาบ้าน (KNN)")
st.caption("โมเดลทำนายราคาบ้านอย่างง่ายด้วย K-Nearest Neighbors")

st.sidebar.header("⚙️ ตั้งค่าโมเดล")
k_value = st.sidebar.slider("จำนวนเพื่อนบ้าน (K)", 1, 30, 5)

# ========== 3. เทรนโมเดล ==========
model = KNeighborsRegressor(n_neighbors=k_value, weights='distance')
model.fit(X_train_scaled, y_train)

# คำนวณความแม่นยำรวม
y_pred_test = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred_test)

# ========== 4. ส่วนกรอกข้อมูล (Compact) ==========
st.markdown("---")
st.subheader("📝 กรอกข้อมูลบ้าน")

col1, col2 = st.columns(2)
with col1:
    med_inc = st.slider("รายได้เฉลี่ยต่อครัวเรือน ($10k)", 0.5, 15.0, 3.5)
    house_age = st.slider("อายุบ้าน (ปี)", 1, 52, 29)
    ave_rooms = st.slider("จำนวนห้องเฉลี่ย", 1.0, 10.0, 5.0)
    ave_beds = st.slider("จำนวนห้องนอนเฉลี่ย", 1.0, 5.0, 1.0)

with col2:
    population = st.slider("จำนวนประชากรในเขต", 3, 35000, 1000)
    ave_occup = st.slider("จำนวนคนต่อครัวเรือน", 1.0, 6.0, 3.0)
    latitude = st.slider("ละติจูด (Latitude)", 32.5, 42.0, 37.5)
    longitude = st.slider("ลองจิจูด (Longitude)", -124.5, -114.0, -119.5)

# ========== 5. ทำนายผล ==========
# เรียงลำดับ Feature ให้ตรงกับตอนเทรน
input_data = pd.DataFrame([[
    med_inc, house_age, ave_rooms, ave_beds, 
    population, ave_occup, latitude, longitude
]], columns=X.columns)

input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0] 
price_usd = prediction * 100000 # แปลงกลับเป็นดอลลาร์
price_thb = price_usd * 36      # แปลงเป็นบาทโดยประมาณ

# ========== 6. แสดงผลลัพธ์ ==========
st.markdown("---")
st.markdown("### 💰 ผลการประเมินราคา")

# แสดงราคาตัวใหญ่ๆ
st.markdown(f"""
<div style="background: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; border-left: 5px solid #00c9ff;">
    <h2 style="margin:0; color:#1f77b4;">${price_usd:,.0f}</h2>
    <p style="margin:5px 0 0 0; color:#666;">หรือประมาณ {price_thb:,.0f} บาท</p>
</div>
""", unsafe_allow_html=True)

st.metric("ความแม่นยำของโมเดล (R² Score)", f"{r2:.2%}", help="ยิ่งใกล้ 100% ยิ่งทำนายได้แม่นยำ")

# กราฟง่ายๆ แสดงการกระจายตัวของราคา
st.markdown("##### 📊 เปรียบเทียบราคาจริง vs ราคาที่ทำนายได้ในชุดข้อมูลทดสอบ")
fig = px.scatter(
    x=y_test[:500], y=y_pred_test[:500], 
    labels={'x': 'ราคาจริง', 'y': 'ราคาที่ทำนาย'},
    opacity=0.6,
    height=300
)
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(fig, use_container_width=True)