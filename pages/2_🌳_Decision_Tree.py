import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import os

st.set_page_config(page_title="Decision Tree - Heart Disease", page_icon="🫀", layout="wide")

# ========== Custom CSS ==========
st.markdown("""
<style>
    .prediction-box {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .risk-high { background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%); }
    .risk-low { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .metric-card {
        background: rgba(255,255,255,0.05);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #38ef7d;
        text-align: center;
    }
    .step-box {
        background: rgba(255,255,255,0.05);
        padding: 10px 15px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 3px solid #00c9ff;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🫀 Decision Tree - Heart Disease Prediction")
st.markdown("### ทำนายความเสี่ยงโรคหัวใจด้วยอัลกอริทึมต้นไม้ตัดสินใจ (Interpretable AI)")

# ========== โหลดข้อมูล ==========
@st.cache_data
def load_data():
    paths = [
        "heart_disease_patient_eda_2000_records2.xlsx",
        "data/heart_disease_patient_eda_2000_records2.xlsx",
        "../heart_disease_patient_eda_2000_records2.xlsx"
    ]
    for path in paths:
        if os.path.exists(path):
            return pd.read_excel(path)
    st.error("❌ ไม่พบไฟล์ `heart_disease_patient_eda_2000_records2.xlsx`")
    return None

df = load_data()
if df is None:
    st.stop()

# ========== Preprocessing ==========
numeric_features = ['age', 'bmi', 'systolic_bp', 'diastolic_bp', 
                    'cholesterol_mg_dl', 'fasting_blood_sugar', 
                    'max_heart_rate', 'risk_score']

categorical_features = ['sex', 'smoking_status', 'exercise_level', 
                        'diabetes', 'family_history', 'chest_pain_type', 
                        'ecg_result']

label_encoders = {}
df_encoded = df.copy()

for col in categorical_features:
    if col in df.columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

feature_cols = numeric_features + [col for col in categorical_features if col in df.columns]
X = df_encoded[feature_cols]
y = df_encoded['heart_disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ========== Sidebar: กรอกข้อมูล ==========
st.sidebar.markdown("## 🏥 กรอกข้อมูลผู้ป่วย")
st.sidebar.markdown("---")

input_data = {}

st.sidebar.markdown("### 🫀 สัญญาณชีพ (Vital Signs)")
input_data['age'] = st.sidebar.slider("อายุ (ปี)", int(df['age'].min()), int(df['age'].max()), int(df['age'].median()))
input_data['max_heart_rate'] = st.sidebar.slider("อัตราการเต้นหัวใจสูงสุด (bpm)", int(df['max_heart_rate'].min()), int(df['max_heart_rate'].max()), int(df['max_heart_rate'].median()))
input_data['systolic_bp'] = st.sidebar.number_input("ความดันโลหิตตัวบน (mmHg)", int(df['systolic_bp'].min()), int(df['systolic_bp'].max()), int(df['systolic_bp'].median()))
input_data['diastolic_bp'] = st.sidebar.number_input("ความดันโลหิตตัวล่าง (mmHg)", int(df['diastolic_bp'].min()), int(df['diastolic_bp'].max()), int(df['diastolic_bp'].median())

st.sidebar.markdown("### 📊 ผลตรวจทางห้องปฏิบัติการ")
input_data['cholesterol_mg_dl'] = st.sidebar.number_input("ระดับคอเลสเตอรอล (mg/dL)", int(df['cholesterol_mg_dl'].min()), int(df['cholesterol_mg_dl'].max()), int(df['cholesterol_mg_dl'].median()))
input_data['fasting_blood_sugar'] = st.sidebar.number_input("ระดับน้ำตาลในเลือด (mg/dL)", int(df['fasting_blood_sugar'].min()), int(df['fasting_blood_sugar'].max()), int(df['fasting_blood_sugar'].median()))
input_data['bmi'] = st.sidebar.slider("ดัชนีมวลกาย (BMI)", float(df['bmi'].min()), float(df['bmi'].max()), float(df['bmi'].median()), 0.1)
input_data['risk_score'] = st.sidebar.slider("คะแนนความเสี่ยงเบื้องต้น", float(df['risk_score'].min()), float(df['risk_score'].max()), float(df['risk_score'].median()), 0.1)

st.sidebar.markdown("### 🏃‍♂️ ไลฟ์สไตล์และประวัติ")
input_data['sex'] = st.sidebar.selectbox("เพศ", df['sex'].unique())
input_data['smoking_status'] = st.sidebar.selectbox("สถานะการสูบบุหรี่", df['smoking_status'].unique())
input_data['exercise_level'] = st.sidebar.selectbox("ระดับการออกกำลังกาย", df['exercise_level'].unique())
input_data['diabetes'] = st.sidebar.selectbox("เป็นเบาหวานหรือไม่", df['diabetes'].unique())
input_data['family_history'] = st.sidebar.selectbox("มีประวัติครอบครัวเป็นโรคหัวใจ", df['family_history'].unique())
input_data['chest_pain_type'] = st.sidebar.selectbox("ลักษณะอาการเจ็บหน้าอก", df['chest_pain_type'].unique())
input_data['ecg_result'] = st.sidebar.selectbox("ผลคลื่นไฟฟ้าหัวใจ (ECG)", df['ecg_result'].unique())

# Encode input
input_encoded = {}
for col in feature_cols:
    if col in numeric_features:
        input_encoded[col] = input_data[col]
    elif col in categorical_features and col in label_encoders:
        try:
            input_encoded[col] = label_encoders[col].transform([str(input_data[col])])[0]
        except:
            input_encoded[col] = 0

# ========== Model Training ==========
@st.cache_resource
def train_tree(X_train, y_train):
    # ใช้ค่า Default ที่ดีเพื่อป้องกัน Overfitting
    model = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, criterion='gini', random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_tree(X_train, y_train)

# ========== Prediction ==========
input_df = pd.DataFrame([input_encoded])
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0]

# ========== แสดงผลทำนาย และ ลูกเล่น ECG ==========
st.markdown("---")
st.markdown("## 🎯 ผลการวิเคราะห์ความเสี่ยง")

col1, col2 = st.columns([1, 1])

with col1:
    if prediction == 1:
        risk_level = "มีความเสี่ยงสูง"
        risk_class = "risk-high"
        emoji = "⚠️"
        ecg_color = "#ff416c"
    else:
        risk_level = "ความเสี่ยงต่ำ"
        risk_class = "risk-low"
        emoji = "✅"
        ecg_color = "#38ef7d"

    st.markdown(f"""
    <div class="prediction-box {risk_class}">
        <div style="font-size:1.5em;">{emoji} {risk_level}</div>
        <div style="font-size:2.5em; font-weight:bold; margin:10px 0;">
            {'ตรวจพบความเสี่ยงโรคหัวใจ' if prediction == 1 else 'ไม่พบความเสี่ยงโรคหัวใจ'}
        </div>
        <div style="font-size:1.2em;">ความน่าจะเป็น: {prediction_proba[prediction]*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # Gauge Chart แสดง Heart Rate
    hr = input_data['max_heart_rate']
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=hr,
        title={'text': "อัตราการเต้นหัวใจ (BPM)"},
        gauge={
            'axis': {'range': [40, 200]},
            'bar': {'color': ecg_color},
            'steps': [
                {'range': [40, 100], 'color': "rgba(56, 239, 125, 0.3)"},
                {'range': [100, 140], 'color': "rgba(255, 165, 0, 0.3)"},
                {'range': [140, 200], 'color': "rgba(255, 65, 108, 0.3)"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 140
            }
        }
    ))
    fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col2:
    # 🎨 ลูกเล่น: เส้นกราฟ ECG จำลอง
    st.markdown(f"###### 📈 จำลองคลื่นไฟฟ้าหัวใจ (Simulated ECG Rhythm)")
    
    # สร้างข้อมูลเส้น ECG ง่ายๆ (Sine wave with spikes)
    t = np.linspace(0, 4, 500) # 4 วินาที
    freq = hr / 60 # ความถี่ต่อวินาที
    ecg_signal = np.zeros_like(t)
    
    beats = int(4 * freq)
    for i in range(beats):
        peak_pos = i / freq + 0.2
        # QRS complex (จังหวะหัวใจเต้น)
        ecg_signal += 1.2 * np.exp(-800 * (t - peak_pos)**2)
        # T wave
        ecg_signal += 0.3 * np.exp(-100 * (t - peak_pos - 0.15)**2)
        # P wave
        ecg_signal += 0.2 * np.exp(-100 * (t - peak_pos + 0.1)**2)

    fig_ecg = go.Figure()
    fig_ecg.add_trace(go.Scatter(
        x=t, y=ecg_signal,
        mode='lines',
        line=dict(color=ecg_color, width=2),
        name='ECG Signal'
    ))
    fig_ecg.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 1.5]),
        plot_bgcolor='rgba(0,0,0,0.05)'
    )
    st.plotly_chart(fig_ecg, use_container_width=True)
    
    st.info(f"💡 **คำแนะนำ:** อัตราการเต้นหัวใจ **{hr} BPM** อยู่ในระดับ " + 
            ("ปกติ" if hr < 100 else "สูง" if hr < 140 else "สูงมาก") + 
            " สำหรับผู้ใหญ่วัย " + str(input_data['age']) + " ปี")

# ========== Decision Path (แบบอ่านง่าย) ==========
st.markdown("### 🛤️ เส้นทางที่โมเดลใช้ตัดสินใจ (Decision Path)")
st.markdown("โมเดลตรวจสอบเงื่อนไขตามลำดับดังนี้:")

node_indicator = model.decision_path(input_df)
feature_arr = model.tree_.feature
threshold_arr = model.tree_.threshold
node_index = node_indicator.indices[node_indicator.nonzero()]

for i, node_id in enumerate(node_index):
    if node_id == node_index[-1]:
        st.markdown(f"""
        <div class="step-box" style="border-left-color: {ecg_color}; background: rgba(255,255,255,0.1);">
            <b>🎯 ขั้นตอนที่ {i+1} (Leaf Node):</b> 
            สรุปผลเป็น <b>{'ความเสี่ยงสูง (Disease)' if prediction == 1 else 'ความเสี่ยงต่ำ (No Disease)'}</b>
        </div>
        """, unsafe_allow_html=True)
    else:
        feature_name = feature_cols[feature_arr[node_id]]
        # แปลงชื่อ feature กลับให้อ่านง่าย
        display_name = feature_name.replace('_', ' ').title()
        threshold = threshold_arr[node_id]
        value = input_df[feature_name].values[0]
        
        if value <= threshold:
            st.markdown(f"""
            <div class="step-box">
                <b>✅ ขั้นตอนที่ {i+1}:</b> <code>{display_name}</code> ≤ {threshold:.2f} 
                <span style="color:#aaa;">(ค่าของคุณ: {value:.2f})</span> → <i>ไปทางซ้าย</i>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="step-box">
                <b>❌ ขั้นตอนที่ {i+1}:</b> <code>{display_name}</code> > {threshold:.2f} 
                <span style="color:#aaa;">(ค่าของคุณ: {value:.2f})</span> → <i>ไปทางขวา</i>
            </div>
            """, unsafe_allow_html=True)

# ========== Model Evaluation (Collapse เพื่อความสะอาด) ==========
with st.expander("📊 ดูประสิทธิภาพของโมเดล (Model Evaluation)"):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
    col2.metric("AUC-ROC", f"{auc(roc_curve(y_test, y_pred_proba)[0], roc_curve(y_test, y_pred_proba)[1]):.2%}")
    col3.metric("จำนวน Features", len(feature_cols))
    
    c1, c2 = st.columns(2)
    with c1:
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        im = ax_cm.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax_cm.set_title('Confusion Matrix')
        ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(['No Disease', 'Disease'])
        ax_cm.set_yticklabels(['No Disease', 'Disease'])
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
        st.pyplot(fig_cm)
    
    with c2:
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {auc(fpr, tpr):.3f}', line=dict(color='#38ef7d', width=3)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='gray', dash='dash')))
        fig_roc.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', height=300)
        st.plotly_chart(fig_roc, use_container_width=True)

# ========== Feature Importance ==========
st.markdown("### 🏆 ปัจจัยที่มีผลต่อการทำนายมากที่สุด (Feature Importance)")
importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values('Importance', ascending=True)

fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h', 
                 color='Importance', color_continuous_scale='RdYlGn_r')
fig_imp.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20))
st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("---")
st.caption("🫀 Decision Tree Heart Disease Predictor | Machine Learning Explorer")