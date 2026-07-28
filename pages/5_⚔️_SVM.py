"""
=============================================================
  Teen Depression Prediction - Streamlit Web App
  รันผ่าน CMD: streamlit run app.py
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(
    page_title="🧠 Teen Depression Predictor",
    page_icon="🧠",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-safe {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        font-size: 1.5rem;
        font-weight: bold;
        color: #155724;
        text-align: center;
    }
    .risk-depression {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        font-size: 1.5rem;
        font-weight: bold;
        color: #721c24;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ==================== Load & Prepare Data ====================
@st.cache_data
def load_and_prepare_data():
    try:
        # ลองอ่านไฟล์โดยไม่มี header ก่อน เพราะข้อมูลตัวอย่างที่ให้เป็น raw data
        df = pd.read_csv("Teen_Mental_Health_Dataset.csv", header=None)
        
        # ตรวจสอบว่าแถวแรกเป็นข้อมูลหรือไม่ (ถ้าใช่ ให้กำหนดชื่อคอลัมน์)
        if str(df.iloc[0, 0]).isdigit() or str(df.iloc[0, 0]) in ['13', '14', '15', '16', '17', '18', '19']:
            df.columns = [
                'age', 'gender', 'daily_social_media_hours', 'platform_usage', 
                'sleep_hours', 'screen_time_before_sleep', 'academic_performance', 
                'physical_activity', 'social_interaction_level', 'stress_level', 
                'anxiety_level', 'addiction_level', 'depression'
            ]
            
        # แปลงประเภทข้อมูล
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df['depression'] = pd.to_numeric(df['depression'], errors='coerce')
        df = df.dropna() # ลบแถวที่มีค่าว่าง
        
        return df
    except Exception as e:
        st.error(f"❌ ไม่สามารถโหลดไฟล์ Teen_Mental_Health_Dataset.csv ได้: {e}")
        return None

df = load_and_prepare_data()

if df is None:
    st.stop()

# ==================== Preprocessing ====================
target = 'depression'
X = df.drop(columns=[target])
y = df[target]

categorical_cols = ['gender', 'platform_usage', 'social_interaction_level']
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# Encode Categorical
label_encoders = {}
X_encoded = X.copy()
for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
    label_encoders[col] = le

# Scale Numerical
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Train/Test Split (สำหรับแสดง Metrics)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ==================== Train Model ====================
@st.cache_resource
def train_svm_model(X_tr, y_tr):
    model = SVC(kernel='rbf', probability=True, random_state=42, C=1.0)
    model.fit(X_tr, y_tr)
    return model

model = train_svm_model(X_train, y_train)

# Evaluate for Sidebar info
y_pred_test = model.predict(X_test)
acc = accuracy_score(y_test, y_pred_test)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/brain.png", width=80)
    st.title("📋 ข้อมูลโมเดล")
    st.markdown("---")
    st.markdown("**🔧 Algorithm:** SVM (Support Vector Machine)")
    st.markdown("**⚙️ Kernel:** RBF")
    st.markdown(f"**🎯 Accuracy:** {acc:.2%}")
    st.markdown(f"**📝 Features:** {len(numerical_cols) + len(categorical_cols)}")
    st.markdown(f"**📊 Dataset Size:** {len(df)} records")
    
    st.markdown("---")
    st.warning("⚠️ ผลลัพธ์เป็นการประเมินเบื้องต้นเท่านั้น ไม่สามารถใช้แทนการวินิจฉัยจากแพทย์หรือผู้เชี่ยวชาญได้")

# ==================== MAIN APP ====================
st.markdown('<h1 class="main-header">🧠 Teen Depression Risk Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">ระบบทำนายความเสี่ยงภาวะซึมเศร้าในวัยรุ่น ด้วย SVM Machine Learning</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔮 ทำนายผล", "📊 วิเคราะห์หลายราย", "ℹ️ เกี่ยวกับโมเดล"])

# ====================== TAB 1: Single Prediction ======================
with tab1:
    st.markdown("### 📝 กรอกข้อมูลเพื่อทำนาย")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 👤 ข้อมูลส่วนบุคคล")
        age = st.slider("🎂 อายุ (ปี)", 13, 19, 15)
        gender = st.selectbox("⚧ เพศ", ["male", "female"])
        platform = st.selectbox("📱 แพลตฟอร์มที่ใช้", ["TikTok", "Instagram", "Both"])
        social_interaction = st.selectbox("🤝 ระดับปฏิสัมพันธ์ทางสังคม", ["low", "medium", "high"])

        st.markdown("#### ⏰ พฤติกรรมการใช้")
        social_media_hours = st.slider("📱 ชั่วโมงใช้โซเชียลมีเดีย/วัน", 0.0, 15.0, 5.0, 0.1)
        screen_before_sleep = st.slider("📺 ใช้หน้าจอก่อนนอน (ชม.)", 0.0, 5.0, 1.5, 0.1)
        sleep_hours = st.slider("😴 ชั่วโมงการนอน/วัน", 3.0, 12.0, 7.0, 0.1)
        physical_activity = st.slider("🏃 กิจกรรมทางกาย (ชม./วัน)", 0.0, 5.0, 1.0, 0.1)

    with col2:
        st.markdown("#### 📈 ผลการเรียนและอารมณ์")
        academic_perf = st.slider("📚 ผลการเรียน (GPA/คะแนน 1-5)", 1.0, 5.0, 3.0, 0.1)

        st.markdown("#### 💭 ระดับความเครียดและอารมณ์ (1-10)")
        stress_level = st.slider("😰 ระดับความเครียด", 1, 10, 5)
        anxiety_level = st.slider("😟 ระดับความกังวล", 1, 10, 5)
        addiction_level = st.slider("📱 ระดับการติดโซเชียล", 1, 10, 5)

    st.markdown("---")

    if st.button("🔮 ทำนายผล", type="primary", use_container_width=True):
        # สร้าง DataFrame จาก input
        input_data = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "daily_social_media_hours": social_media_hours,
            "platform_usage": platform,
            "sleep_hours": sleep_hours,
            "screen_time_before_sleep": screen_before_sleep,
            "academic_performance": academic_perf,
            "physical_activity": physical_activity,
            "social_interaction_level": social_interaction,
            "stress_level": stress_level,
            "anxiety_level": anxiety_level,
            "addiction_level": addiction_level,
        }])

        # Preprocess input
        input_processed = input_data.copy()
        for col, le in label_encoders.items():
            if col in input_processed.columns:
                # ใช้ try-except เพื่อป้องกันกรณีค่าที่ไม่เคยเห็นตอน train
                try:
                    input_processed[col] = le.transform(input_processed[col].astype(str))
                except ValueError:
                    input_processed[col] = 0  # Fallback to first class

        input_scaled = scaler.transform(input_processed)

        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        st.markdown("### 🎯 ผลการทำนาย")

        if prediction == 1:
            st.markdown(
                '<div class="risk-depression">🔴 มีความเสี่ยงภาวะซึมเศร้า (Depression Risk)</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="risk-safe">🟢 ไม่มีความเสี่ยงภาวะซึมเศร้า (No Depression Risk)</div>',
                unsafe_allow_html=True
            )

        # Probability Bar Chart
        st.markdown("#### 📊 ความน่าจะเป็น")
        prob_df = pd.DataFrame({
            "Class": ["No Depression (0)", "Depression (1)"],
            "Probability": probability
        })
        fig = px.bar(
            prob_df, x="Class", y="Probability",
            color="Class",
            color_discrete_map={"No Depression (0)": "#28a745", "Depression (1)": "#dc3545"},
            text_auto=".1%"
        )
        fig.update_layout(yaxis_range=[0, 1], showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Radar Chart
        st.markdown("#### 🕸️ Radar Chart - ภาพรวมปัจจัย")
        radar_data = {
            "ปัจจัย": [
                "Social Media (Hrs)", "Screen Before Sleep", "Sleep (Hrs)",
                "Physical Activity", "Academic Perf",
                "Stress", "Anxiety", "Addiction"
            ],
            "คะแนน": [
                social_media_hours / 15.0, screen_before_sleep / 5.0, sleep_hours / 12.0,
                physical_activity / 5.0, academic_perf / 5.0,
                stress_level / 10.0, anxiety_level / 10.0, addiction_level / 10.0
            ]
        }
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_data["คะแนน"] + [radar_data["คะแนน"][0]],
            theta=radar_data["ปัจจัย"] + [radar_data["ปัจจัย"][0]],
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.3)",
            line=dict(color="#1f77b4")
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
        st.plotly_chart(fig_radar, use_container_width=True)

        # คำแนะนำ
        st.markdown("### 💡 คำแนะนำ")
        if prediction == 1:
            st.warning("🚨 **ควรพบผู้เชี่ยวชาญด้านสุขภาพจิตเพื่อประเมินเพิ่มเติม**")
            st.markdown("""
            - ลดเวลาใช้โซเชียลมีเดียลงอย่างจริงจัง โดยเฉพาะก่อนนอน
            - เพิ่มชั่วโมงการนอนหลับให้ได้อย่างน้อย 7-8 ชม.
            - ออกกำลังกายหรือทำกิจกรรมทางกายสม่ำเสมอ
            - พูดคุยกับคนที่ไว้ใจเกี่ยวกับความรู้สึก
            - **สายด่วนสุขภาพจิต: 1323 (กรมสุขภาพจิต)**
            """)
        else:
            st.success("✅ สุขภาพจิตอยู่ในเกณฑ์ดี รักษากิจวัตรที่ดีไว้")
            st.markdown("""
            - หมั่นออกกำลังกายและนอนหลับให้เพียงพอ
            - รักษาความสัมพันธ์ที่ดีกับครอบครัวและเพื่อน
            - ใช้โซเชียลมีเดียอย่างมีสติและกำหนดเวลาใช้งาน
            """)

# ====================== TAB 2: Batch Prediction ======================
with tab2:
    st.markdown("### 📊 อัปโหลดไฟล์ CSV เพื่อทำนายหลายราย")

    st.markdown("""
    **รูปแบบ CSV ที่ต้องมีคอลัมน์ (ตามลำดับหรือมี Header ตรงกัน):**
    `age, gender, daily_social_media_hours, platform_usage, sleep_hours, screen_time_before_sleep, academic_performance, physical_activity, social_interaction_level, stress_level, anxiety_level, addiction_level`
    """)

    uploaded_file = st.file_uploader("📁 เลือกไฟล์ CSV", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.markdown("#### 📋 ตัวอย่างข้อมูลที่อัปโหลด")
        st.dataframe(batch_df.head(), use_container_width=True)

        if st.button("🔮 ทำนายทั้งหมด", type="primary"):
            try:
                batch_processed = batch_df.copy()
                
                # Ensure columns match
                for col in categorical_cols:
                    if col in batch_processed.columns:
                        try:
                            batch_processed[col] = label_encoders[col].transform(batch_processed[col].astype(str))
                        except ValueError:
                            batch_processed[col] = 0
                
                # Fill missing columns with 0 if any
                for col in X_encoded.columns:
                    if col not in batch_processed.columns:
                        batch_processed[col] = 0
                        
                batch_processed = batch_processed[X_encoded.columns]
                batch_scaled = scaler.transform(batch_processed)
                
                predictions = model.predict(batch_scaled)
                probabilities = model.predict_proba(batch_scaled)

                batch_df["Predicted_Label"] = predictions
                batch_df["Predicted_Class"] = np.where(predictions == 1, "Depression", "No Depression")
                batch_df["Depression_Probability"] = probabilities[:, 1]

                st.success(f"✅ ทำนายสำเร็จ {len(batch_df)} ราย")
                st.dataframe(batch_df, use_container_width=True)

                # สรุป
                st.markdown("#### 📊 สรุปผลการทำนาย")
                summary = batch_df["Predicted_Class"].value_counts().reset_index()
                summary.columns = ["Class", "Count"]

                fig_pie = px.pie(
                    summary, values="Count", names="Class",
                    color="Class",
                    color_discrete_map={"No Depression": "#28a745", "Depression": "#dc3545"},
                    hole=0.4
                )
                st.plotly_chart(fig_pie, use_container_width=True)

                csv_result = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 ดาวน์โหลดผลทำนาย (CSV)",
                    data=csv_result,
                    file_name="teen_mental_health_predictions.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาด: {e}")
                st.exception(e)

# ====================== TAB 3: About Model ======================
with tab3:
    st.markdown("### ℹ️ เกี่ยวกับโมเดล")

    st.markdown("""
    #### 🧠 โมเดล SVM (Support Vector Machine)

    **SVM** คืออัลกอริทึม Machine Learning ที่ใช้สำหรับ Binary Classification
    โดยค้นหา Hyperplane ที่แบ่งกลุ่มข้อมูลได้ดีที่สุด เหมาะสำหรับการแยกแยะรูปแบบที่ซับซ้อนในข้อมูลสุขภาพจิต

    #### 🎯 Target Variable
    - **0** = ไม่มีความเสี่ยงภาวะซึมเศร้า (No Depression)
    - **1** = มีความเสี่ยงภาวะซึมเศร้า (Depression)

    #### ⚙️ กระบวนการทำงาน

    | ขั้นตอน | รายละเอียด |
    |---------|------------|
    | 1. Data Loading | โหลดข้อมูล CSV |
    | 2. Preprocessing | Encode categorical (เช่น เพศ, แพลตฟอร์ม), Scale numerical |
    | 3. Train/Test Split | แบ่งข้อมูล 80/20 |
    | 4. Training | เทรน SVM ด้วย Kernel RBF |
    | 5. Evaluation | Accuracy, Classification Report |
    """)

    st.markdown("#### 📊 Classification Report (จากชุดข้อมูลทดสอบ)")
    report = classification_report(y_test, y_pred_test, target_names=["No Depression (0)", "Depression (1)"], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #999;'>"
    "🧠 Teen Depression Risk Predictor | Built with SVM + Streamlit | "
    "⚠️ สำหรับการศึกษาเท่านั้น"
    "</div>",
    unsafe_allow_html=True
)