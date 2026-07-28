"""
============================================================
Heart Disease Prediction Web App (Custom Dataset)
รันด้วยคำสั่ง: streamlit run app.py
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score

# ==================== Configuration ====================
DEVELOPER_NAME = "pxrawit"
DEVELOPER_EMAIL = "puwaritchammunkung@gmail.com"
DATASET_PATH = "heart_disease_patient_eda_2000_records2.xlsx"

# ==================== Page Configuration ====================
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Custom CSS ====================
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    .main-title { font-size: 2.5rem; font-weight: 700; color: #2c3e50; text-align: center; margin-bottom: 0.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }
    .sub-title { font-size: 1.1rem; color: #7f8c8d; text-align: center; margin-bottom: 2rem; }
    .result-safe { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 2rem; border-radius: 15px; text-align: center; font-size: 1.3rem; font-weight: 600; box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3); }
    .result-danger { background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); color: white; padding: 2rem; border-radius: 15px; text-align: center; font-size: 1.3rem; font-weight: 600; box-shadow: 0 4px 15px rgba(235, 51, 73, 0.3); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%); }
    [data-testid="stSidebar"] * { color: white !important; }
    .stButton > button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; font-weight: 600; font-size: 1.1rem; padding: 0.6rem 2rem; border-radius: 10px; border: none; transition: all 0.3s ease; }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4); }
    .email-card { background: rgba(255, 255, 255, 0.1); padding: 1.2rem; border-radius: 10px; text-align: center; backdrop-filter: blur(10px); }
    .email-link { color: #ecf0f1 !important; text-decoration: none; font-size: 1rem; font-weight: 500; word-break: break-all; }
    .email-link:hover { color: #3498db !important; }
    .dev-profile { text-align: center; padding: 1rem; background: rgba(255, 255, 255, 0.05); border-radius: 10px; margin-bottom: 1rem; }
    .dev-name { font-size: 1.2rem; font-weight: 700; color: #ecf0f1 !important; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ==================== Load Data & Train Model ====================
@st.cache_resource
def load_and_train():
    if not os.path.exists(DATASET_PATH):
        st.error(f"❌ ไม่พบไฟล์ {DATASET_PATH} กรุณาวางไฟล์ไว้ในโฟลเดอร์เดียวกัน")
        st.stop()
    
    df = pd.read_excel(DATASET_PATH)
    
    # แยก Features และ Target
    target_col = 'heart_disease'
    numeric_cols = ['age', 'bmi', 'systolic_bp', 'diastolic_bp', 'cholesterol_mg_dl', 'fasting_blood_sugar', 'max_heart_rate']
    categorical_cols = ['sex', 'smoking_status', 'exercise_level', 'diabetes', 'family_history', 'chest_pain_type', 'ecg_result']
    
    # Encode Categorical Data
    encoders = {}
    df_encoded = df.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        
    feature_cols = numeric_cols + categorical_cols
    X = df_encoded[feature_cols]
    y = df_encoded[target_col]
    
    # Train Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Metrics
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    
    return model, encoders, feature_cols, numeric_cols, categorical_cols, metrics

try:
    model, encoders, feature_cols, numeric_cols, categorical_cols, metrics = load_and_train()
except Exception as e:
    st.error(f"❌ เกิดข้อผิดพลาด: {e}")
    st.stop()

# ==================== Sidebar ====================
with st.sidebar:
    st.markdown(f"""
        <div class="dev-profile">
            <div style="font-size: 3rem;">👨‍💻</div>
            <div class="dev-name">{DEVELOPER_NAME}</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 โมเดล Information")
    st.info(f"""
    **Algorithm:** Decision Tree  
    **Accuracy:** {metrics['accuracy']:.2%}  
    **ROC-AUC:** {metrics['roc_auc']:.3f}  
    **Dataset:** 2,000 Records
    """)
    
# ==================== Main Content ====================
st.markdown('<p class="main-title">🫀 ระบบทำนายความเสี่ยงโรคหัวใจ</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Heart Disease Risk Prediction using Decision Tree ML Model</p>', unsafe_allow_html=True)

st.markdown("### 📝 กรุณากรอกข้อมูลสุขภาพ")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 👤 ข้อมูลพื้นฐาน")
    age = st.number_input("🎂 อายุ (Age)", min_value=18, max_value=100, value=55, step=1)
    bmi = st.number_input("⚖️ ดัชนีมวลกาย (BMI)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    
    sex = st.selectbox("⚧ เพศ (Sex)", options=["Male", "Female"])
    smoking_status = st.selectbox("🚬 สถานะการสูบบุหรี่", options=["Never", "Former", "Current"])
    exercise_level = st.selectbox("🏃 ระดับการออกกำลังกาย", options=["Low", "Moderate", "High"])
    
    systolic_bp = st.number_input("💉 ความดันโลหิตตัวบน (Systolic BP) [mm Hg]", min_value=80, max_value=250, value=120, step=1)
    diastolic_bp = st.number_input("💉 ความดันโลหิตตัวล่าง (Diastolic BP) [mm Hg]", min_value=40, max_value=150, value=80, step=1)

with col2:
    st.markdown("#### 🏥 ผลการตรวจและประวัติ")
    cholesterol = st.number_input("🩸 คอเลสเตอรอล (Cholesterol) [mg/dl]", min_value=100, max_value=600, value=200, step=1)
    fasting_blood_sugar = st.number_input("🍬 น้ำตาลในเลือดขณะอดอาหาร [mg/dl]", min_value=50, max_value=300, value=100, step=1)
    max_heart_rate = st.number_input("💓 อัตราการเต้นหัวใจสูงสุด (Max HR) [bpm]", min_value=60, max_value=220, value=150, step=1)
    
    diabetes = st.selectbox("🩸 เป็นเบาหวานหรือไม่", options=["FALSE", "TRUE"])
    family_history = st.selectbox("🧬 มีประวัติครอบครัวเป็นโรคหัวใจ", options=["FALSE", "TRUE"])
    chest_pain_type = st.selectbox("💔 ประเภทอาการเจ็บหน้าอก", options=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    ecg_result = st.selectbox("📈 ผล ECG", options=["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])

# ==================== Prediction Button ====================
st.markdown("---")
predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
with predict_col2:
    predict_clicked = st.button("🔮 ทำนายผล (Predict)", use_container_width=True, type="primary")

# ==================== Prediction Result ====================
if predict_clicked:
    # สร้าง DataFrame จากข้อมูลที่ผู้ใช้กรอก
    input_dict = {
        'age': age, 'bmi': bmi, 'systolic_bp': systolic_bp, 'diastolic_bp': diastolic_bp,
        'cholesterol_mg_dl': cholesterol, 'fasting_blood_sugar': fasting_blood_sugar,
        'max_heart_rate': max_heart_rate, 'sex': sex, 'smoking_status': smoking_status,
        'exercise_level': exercise_level, 'diabetes': diabetes, 'family_history': family_history,
        'chest_pain_type': chest_pain_type, 'ecg_result': ecg_result
    }
    input_df = pd.DataFrame([input_dict])
    
    # Encode ข้อมูลให้ตรงกับตอนฝึกโมเดล
    for col in categorical_cols:
        # ถ้ามีค่าที่ไม่เคยเห็นตอนฝึก ให้ใช้ค่าแรกสุดแทน (ป้องกัน Error)
        try:
            input_df[col] = encoders[col].transform(input_df[col].astype(str))
        except ValueError:
            input_df[col] = 0 
            
    # ทำนายผล
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    
    st.markdown("---")
    st.markdown("### 🎯 ผลการทำนาย")
    
    if prediction == 1:
        risk_prob = probability[1] * 100
        st.markdown(f'<div class="result-danger">⚠️ <strong>มีความเสี่ยงเป็นโรคหัวใจ</strong><br><span style="font-size: 2rem;">ความน่าจะเป็น: {risk_prob:.1f}%</span></div>', unsafe_allow_html=True)
    else:
        safe_prob = probability[0] * 100
        st.markdown(f'<div class="result-safe">✅ <strong>ไม่พบความเสี่ยงโรคหัวใจ</strong><br><span style="font-size: 2rem;">ความน่าจะเป็น: {safe_prob:.1f}%</span></div>', unsafe_allow_html=True)
    
    # Gauge Chart
    st.markdown("#### 📊 Probability Distribution")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability[1] * 100,
        title={'text': "ความเสี่ยงโรคหัวใจ (%)", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': '#38ef7d'},
                {'range': [30, 60], 'color': '#f39c12'},
                {'range': [60, 100], 'color': '#eb3349'}
            ],
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📋 ดูข้อมูลที่คุณกรอก", expanded=False):
        st.dataframe(input_df, use_container_width=True)
    
    with st.expander("🔍 Feature Importance", expanded=False):
        importances = model.feature_importances_
        feature_imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values('Importance', ascending=False)
        
        fig_imp = go.Figure(data=[go.Bar(x=feature_imp_df['Feature'], y=feature_imp_df['Importance'], marker_color='rgb(102, 126, 234)')])
        fig_imp.update_layout(title='Feature Importance', xaxis_title='Features', yaxis_title='Importance', height=400)
        st.plotly_chart(fig_imp, use_container_width=True)

# ==================== Footer ====================
st.markdown("---")
st.markdown(f"""
    <div style='text-align: center; color: #7f8c8d; padding: 1rem;'>
        <p style='font-size: 0.9rem;'>⚕️ <strong>คำเตือน:</strong> ผลลัพธ์จากการทำนายเป็นเพียงการประเมินเบื้องต้น ไม่สามารถใช้แทนการวินิจฉัยจากแพทย์ได้</p>
        <p style='font-size: 0.85rem; margin-top: 1rem;'>📧 ติดต่อผู้พัฒนา: <a href="mailto:{DEVELOPER_EMAIL}" style="color: #3498db;">{DEVELOPER_EMAIL}</a></p>
        <p style='font-size: 0.8rem; margin-top: 0.5rem;'>© 2026 {DEVELOPER_NAME}</p>
    </div>
""", unsafe_allow_html=True)