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
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .prediction-value { font-size: 2.5em; font-weight: bold; margin: 10px 0; }
    .risk-high { background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%); }
    .risk-low { background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%); }
    .monitor-bg { background-color: #000; border-radius: 10px; padding: 10px; border: 2px solid #333; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🫀 Decision Tree: ทำนายความเสี่ยงโรคหัวใจ")
st.markdown("แบบจำลองที่อธิบายได้ง่าย เหมือนการถาม-ตอบ เพื่อประเมินสุขภาพหัวใจของคุณ")

# ========== โหลดข้อมูล ==========
@st.cache_data
def load_data():
    paths = ["heart_disease_patient_eda_2000_records2.xlsx", "data/heart_disease_patient_eda_2000_records2.xlsx"]
    for path in paths:
        if os.path.exists(path):
            return pd.read_excel(path)
    st.error("❌ ไม่พบไฟล์ heart_disease_patient_eda_2000_records2.xlsx")
    return None

df = load_data()
if df is None:
    st.stop()

# ========== Preprocessing ==========
numeric_features = ['age', 'bmi', 'systolic_bp', 'diastolic_bp', 'cholesterol_mg_dl', 'fasting_blood_sugar', 'max_heart_rate', 'risk_score']
categorical_features = ['sex', 'smoking_status', 'exercise_level', 'diabetes', 'family_history', 'chest_pain_type', 'ecg_result']

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
st.sidebar.markdown("### 🏥 กรอกข้อมูลผู้ป่วย")
st.sidebar.markdown("---")

input_data = {}
st.sidebar.markdown("#### 📊 สัญญาณชีพ")
input_data['age'] = st.sidebar.slider("อายุ (ปี)", int(df['age'].min()), int(df['age'].max()), int(df['age'].median()))
input_data['bmi'] = st.sidebar.slider("ค่า BMI", float(df['bmi'].min()), float(df['bmi'].max()), float(df['bmi'].median()), 0.1)
input_data['max_heart_rate'] = st.sidebar.number_input("อัตราการเต้นหัวใจสูงสุด (bpm)", int(df['max_heart_rate'].min()), int(df['max_heart_rate'].max()), int(df['max_heart_rate'].median()))
input_data['systolic_bp'] = st.sidebar.number_input("ความดันตัวบน (mmHg)", int(df['systolic_bp'].min()), int(df['systolic_bp'].max()), int(df['systolic_bp'].median()))
input_data['cholesterol_mg_dl'] = st.sidebar.number_input("ระดับคอเลสเตอรอล", int(df['cholesterol_mg_dl'].min()), int(df['cholesterol_mg_dl'].max()), int(df['cholesterol_mg_dl'].median()))

st.sidebar.markdown("#### 🏃‍♂️ ประวัติและไลฟ์สไตล์")
input_data['sex'] = st.sidebar.selectbox("เพศ", df['sex'].unique())
input_data['chest_pain_type'] = st.sidebar.selectbox("ลักษณะอาการเจ็บหน้าอก", df['chest_pain_type'].unique())
input_data['smoking_status'] = st.sidebar.selectbox("สถานะการสูบบุหรี่", df['smoking_status'].unique())
input_data['diabetes'] = st.sidebar.selectbox("เป็นเบาหวาน", df['diabetes'].unique())

# Encode สำหรับทำนาย
input_encoded = {}
for col in feature_cols:
    if col in numeric_features:
        input_encoded[col] = input_data.get(col, df[col].median())
    elif col in categorical_features and col in label_encoders:
        try:
            input_encoded[col] = label_encoders[col].transform([str(input_data[col])])[0]
        except:
            input_encoded[col] = 0

# เติมค่า default ให้ครบทุก feature หากไม่มีใน sidebar
for col in feature_cols:
    if col not in input_encoded:
        input_encoded[col] = df_encoded[col].mode()[0]

# ========== เทรนโมเดล ==========
@st.cache_resource
def train_tree(X_tr, y_tr):
    model = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, criterion='gini', random_state=42)
    model.fit(X_tr, y_tr)
    return model

model = train_tree(X_train, y_train)

# ========== ทำนายผล ==========
input_df = pd.DataFrame([input_encoded])
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0]

# กำหนดสีและข้อความตามผลทำนาย
if prediction == 1:
    risk_level, risk_class, emoji, ecg_color = "มีความเสี่ยงสูง", "risk-high", "⚠️", "#ff416c"
    result_text = "ควรปรึกษาแพทย์เพื่อตรวจวินิจฉัยเพิ่มเติม"
else:
    risk_level, risk_class, emoji, ecg_color = "ความเสี่ยงต่ำ", "risk-low", "✅", "#56ab2f"
    result_text = "สุขภาพหัวใจอยู่ในเกณฑ์ที่ดี รักษาพฤติกรรมนี้ต่อไป"

# ========== ส่วนแสดงผลหลัก (Hero Section) ==========
col1, col2 = st.columns([1, 1.5])

with col1:
    st.markdown(f"""
    <div class="prediction-box {risk_class}">
        <div style="font-size:1.3em;">{emoji} ผลการประเมิน: {risk_level}</div>
        <div class="prediction-value">{'ตรวจพบความเสี่ยง' if prediction == 1 else 'ไม่พบความเสี่ยง'}</div>
        <div style="font-size:1.1em; margin-top:10px;">ความน่าจะเป็น: <b>{prediction_proba[prediction]*100:.1f}%</b></div>
        <div style="font-size:0.9em; margin-top:15px; opacity:0.9;">{result_text}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # แสดง Feature Importance แบบย่อ
    importances = model.feature_importances_
    top_features = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values('Importance', ascending=False).head(3)
    st.markdown("##### 🔑 ปัจจัยที่มีผลต่อการตัดสินใจมากที่สุด:")
    for _, row in top_features.iterrows():
        st.progress(float(row['Importance']))
        st.caption(f"{row['Feature'].replace('_', ' ').title()} ({row['Importance']:.1%})")

with col2:
    st.markdown("##### 📟 เครื่องตรวจคลื่นไฟฟ้าหัวใจ (Simulated ECG)")
    # สร้างเส้น ECG จำลอง
    bpm = input_data['max_heart_rate']
    duration = 4 # วินาที
    t = np.linspace(0, duration, 400)
    freq = bpm / 60.0
    signal = np.zeros_like(t)
    
    # สร้างคลื่น P, QRS, T จำลอง
    for i in range(int(freq * duration) + 1):
        peak_time = i / freq + 0.2
        signal += 1.2 * np.exp(-600 * (t - peak_time)**2)  # QRS (หัวใจบีบตัว)
        signal += 0.25 * np.exp(-80 * (t - peak_time - 0.15)**2) # T wave
        signal += 0.15 * np.exp(-80 * (t - peak_time + 0.1)**2)  # P wave

    fig_ecg = go.Figure()
    fig_ecg.add_trace(go.Scatter(x=t, y=signal, mode='lines', line=dict(color=ecg_color, width=2.5), name='ECG'))
    fig_ecg.update_layout(
        plot_bgcolor='rgba(0,0,0,0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', showticklabels=False, zeroline=False, range=[-0.5, 1.5]),
        height=220,
        margin=dict(l=0, r=0, t=0, b=0),
        annotations=[dict(x=0.02, y=1.3, xref='paper', yref='paper', text=f"HR: {bpm} BPM", showarrow=False, font=dict(color=ecg_color, size=16, family="monospace"))]
    )
    st.plotly_chart(fig_ecg, use_container_width=True)

# ========== Decision Path (อธิบายเหตุผล) ==========
st.markdown("---")
st.markdown("### 🛤️ ทำไมโมเดลถึงตัดสินแบบนี้? (Decision Path)")
st.markdown("ต้นไม้ตัดสินใจตรวจสอบเงื่อนไขทีละขั้น ดังนี้:")

node_indicator = model.decision_path(input_df)
feature_arr = model.tree_.feature
threshold_arr = model.tree_.threshold
node_index = node_indicator.indices[node_indicator.nonzero()]

path_cols = st.columns(len(node_index))
for i, node_id in enumerate(node_index):
    with path_cols[i]:
        if node_id == node_index[-1]:
            st.markdown(f"""
            <div style="background:{ecg_color}; color:white; padding:15px; border-radius:10px; text-align:center; height:100%;">
                <b>🎯 สรุปผล</b><br>
                {'ความเสี่ยงสูง' if prediction == 1 else 'ความเสี่ยงต่ำ'}
            </div>
            """, unsafe_allow_html=True)
        else:
            feature_name = feature_cols[feature_arr[node_id]].replace('_', ' ').title()
            threshold = threshold_arr[node_id]
            value = input_df[feature_name.replace(' ', '_').lower()].values[0] # Hack to match col name
            
            # Find original col name
            orig_col = [c for c in feature_cols if c.replace('_',' ').title() == feature_name][0]
            val = input_encoded[orig_col]
            
            if val <= threshold:
                st.markdown(f"""
                <div style="background:#e3f2fd; padding:15px; border-radius:10px; border-left:4px solid #2196f3; height:100%;">
                    <b>ขั้นที่ {i+1}</b><br>
                    <code>{feature_name} ≤ {threshold:.1f}</code><br>
                    <span style="color:#666; font-size:0.9em;">(ค่าของคุณ: {val})</span> ✅
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:#ffebee; padding:15px; border-radius:10px; border-left:4px solid #f44336; height:100%;">
                    <b>ขั้นที่ {i+1}</b><br>
                    <code>{feature_name} > {threshold:.1f}</code><br>
                    <span style="color:#666; font-size:0.9em;">(ค่าของคุณ: {val})</span> ❌
                </div>
                """, unsafe_allow_html=True)

# ========== ส่วนแสดงผลประสิทธิภาพโมเดล (ซ่อนใน Expander เพื่อให้หน้าจอดูสะอาด) ==========
with st.expander("📊 ดูประสิทธิภาพเชิงเทคนิคของโมเดล (สำหรับ Developer/Data Scientist)"):
    st.markdown("#### 1. Model Metrics")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
    m_col2.metric("AUC-ROC", f"{auc(roc_curve(y_test, y_pred_proba)[0], roc_curve(y_test, y_pred_proba)[1]):.2%}")
    
    st.markdown("#### 2. Confusion Matrix & ROC Curve")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    im = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.set_title('Confusion Matrix')
    ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['No Disease', 'Disease']); ax1.set_yticklabels(['No Disease', 'Disease'])
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    ax2.plot(fpr, tpr, color='#56ab2f', lw=2, label=f'ROC curve (AUC = {auc(fpr, tpr):.2f})')
    ax2.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    ax2.set_title('ROC Curve')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.legend(loc="lower right")
    
    st.pyplot(fig)

    st.markdown("#### 3. โครงสร้าง Decision Tree เต็มรูปแบบ")
    fig_tree, ax_tree = plt.subplots(figsize=(20, 10))
    plot_tree(model, feature_names=feature_cols, class_names=['No Disease', 'Disease'], filled=True, rounded=True, fontsize=9, ax=ax_tree)
    st.pyplot(fig_tree)

st.markdown("---")
st.caption("🌳 Decision Tree Heart Disease Predictor | Machine Learning Explorer")