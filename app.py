import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title="ML Models Explorer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== Custom CSS ==========
st.markdown("""
<style>
    .main-title {
        font-size: 3em;
        background: linear-gradient(90deg, #00c9ff, #92fe9d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.2em;
    }
    .profile-container {
        background: linear-gradient(135deg, rgba(0,201,255,0.1), rgba(146,254,157,0.1));
        padding: 30px;
        border-radius: 20px;
        border: 2px solid rgba(0,201,255,0.3);
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .profile-name {
        font-size: 2.2em;
        font-weight: bold;
        color: #00c9ff;
        margin: 0;
    }
    .profile-info {
        font-size: 1.1em;
        color: #ddd;
        margin: 5px 0;
    }
    .profile-badge {
        display: inline-block;
        background: linear-gradient(90deg, #00c9ff, #92fe9d);
        color: #000;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        margin: 5px 5px 5px 0;
        font-size: 0.9em;
    }
    .model-card {
        background: linear-gradient(135deg, rgba(0,201,255,0.1), rgba(146,254,157,0.1));
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
        transition: 0.3s;
        cursor: pointer;
        height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .model-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,201,255,0.3);
    }
    .icon { font-size: 3em; margin-bottom: 10px; }
    .quote-box {
        background: rgba(255,255,255,0.05);
        padding: 20px;
        border-radius: 15px;
        border-left: 4px solid #00c9ff;
        font-style: italic;
        color: #ccc;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ========== Header ==========
st.markdown('<h1 class="main-title">🤖 ML Models Explorer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">สำรวจ 6 อัลกอริทึม Machine Learning พื้นฐาน พร้อม Interactive Demo</p>', unsafe_allow_html=True)

st.markdown("---")

# ========== 👤 ส่วนข้อมูลส่วนตัว ==========
st.markdown("## 👤 เกี่ยวกับผู้จัดทำ")

# ============================================================
# ⚠️ แก้ไขข้อมูลตรงนี้ได้เลย
# ============================================================
STUDENT_NAME = "ภูวฤทธิ์ แช่มมั่นคง"           # ชื่อ-นามสกุล
STUDENT_ID = "664245031"              # รหัสนักศึกษา
STUDENT_SECTION = "หมู่เรียน 66/44"         # หมู่เรียน
STUDENT_PROGRAM = "วิทยาการคอมพิวเตอร์" # สาขา
STUDENT_YEAR = "ชั้นปีที่ 4"            # ชั้นปี

PROFILE_IMAGE_PATH = "profile.jpg"     # path ของรูปโปรไฟล์
# ============================================================

col_img, col_info = st.columns([1, 2])

with col_img:
    # แสดงรูปโปรไฟล์
    try:
        st.image(
            PROFILE_IMAGE_PATH,
            width=300,
            caption="📸 รูปโปรไฟล์"
        )
    except:
        # ถ้าไม่เจอรูป ให้แสดง placeholder
        st.markdown("""
        <div style="text-align:center; padding:50px; background:rgba(255,255,255,0.05); border-radius:15px; border:2px dashed rgba(0,201,255,0.3);">
            <div style="font-size:5em;">👤</div>
            <p style="color:#888;">วางรูป <code>profile.png</code><br>ในโฟลเดอร์เดียวกับ app.py</p>
        </div>
        """, unsafe_allow_html=True)

with col_info:
    st.markdown(f"""
    <div class="profile-container">
        <p class="profile-name">👨‍🎓 {STUDENT_NAME}</p>
        <p class="profile-info">🆔 รหัสนักศึกษา: <b>{STUDENT_ID}</b></p>
        <p class="profile-info">🏫 {STUDENT_SECTION} | สาขา{STUDENT_PROGRAM}</p>
        <p class="profile-info">📚 {STUDENT_YEAR}</p>
        <div style="margin-top:15px;">
            <span class="profile-badge">🤖 Machine Learning</span>
            <span class="profile-badge">🐍 Python</span>
            <span class="profile-badge">📊 Data Science</span>
            <span class="profile-badge">🧠 AI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ========== 📝 แนะนำตัว ==========
st.markdown("### 📝 แนะนำตัว")

st.markdown("""
<div class="quote-box">
    💡 "โปรเจกต์นี้จัดทำขึ้นเพื่อศึกษาอัลกอริทึม Machine Learning พื้นฐาน 6 ชนิด 
    โดยเน้นการทำความเข้าใจผ่านการทำ Interactive Visualization 
    เพื่อให้สามารถเห็นภาพการทำงานของโมเดลแต่ละชนิดได้ชัดเจน"
</div>
""", unsafe_allow_html=True)

# ========== 📊 สถิติโปรเจกต์ ==========
st.markdown("### 📊 สถิติโปรเจกต์")

stat1, stat2, stat3, stat4 = st.columns(4)
stat1.metric("🧠 จำนวนโมเดล", "6", "อัลกอริทึม")
stat2.metric("📈 จำนวน Dataset", "2", "datasets")
stat3.metric("🎨 Visualizations", "20+", "กราฟ")
stat4.metric("📅 สร้างเมื่อ", datetime.now().strftime("%d/%m/%Y"))

st.markdown("---")

# ========== 🎯 เลือกโมเดล ==========
st.markdown("## 🎯 เลือกโมเดลที่ต้องการเรียนรู้")
st.markdown("คลิกที่การ์ดเพื่อไปยังหน้าของโมเดลนั้นๆ")

models_info = [
    ("📍", "KNN", "K-Nearest Neighbors", "ทำนายจากเพื่อนบ้านที่ใกล้ที่สุด"),
    ("🌳", "Decision Tree", "ต้นไม้ตัดสินใจ", "แบ่งข้อมูลด้วยกฎ if-else"),
    ("📈", "Regression", "Linear Regression", "หาเส้นตรงที่ fit ข้อมูลดีที่สุด"),
    ("⚔️", "SVM", "Support Vector Machine", "หาเส้นแบ่งที่กว้างที่สุด"),
    ("🎯", "K-Means", "Clustering", "จัดกลุ่มข้อมูลเป็น K คลัสเตอร์"),
    ("👥", "Ensemble", "Random Forest", "รวมหลายโมเดลเพื่อทำนายแม่นยำขึ้น"),
]

cols = st.columns(3)
for i, (icon, name, full_name, desc) in enumerate(models_info):
    with cols[i % 3]:
        st.markdown(f"""
        <div class="model-card">
            <div class="icon">{icon}</div>
            <h3>{name}</h3>
            <p style="color:#92fe9d; font-size:0.85em; margin:5px 0;">{full_name}</p>
            <p style="color:#bbb; font-size:0.8em;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("👈 **เลือกโมเดลจากเมนูด้านซ้ายมือเพื่อเริ่มเรียนรู้**")

st.markdown("---")

# ========== 🛠️ เทคโนโลยีที่ใช้ ==========
st.markdown("## 🛠️ เทคโนโลยีที่ใช้ในโปรเจกต์")

tech1, tech2, tech3 = st.columns(3)

with tech1:
    st.markdown("""
    **🐍 ภาษาโปรแกรม**
    - Python 3.10+
    - Streamlit Framework
    """)

with tech2:
    st.markdown("""
    **📚 Libraries**
    - scikit-learn
    - pandas & numpy
    - matplotlib & plotly
    """)

with tech3:
    st.markdown("""
    **📊 Datasets**
    - California Housing
    - Heart Disease Patient
    """)

# ========== 📚 เนื้อหาที่เรียนรู้ ==========
st.markdown("## 📚 เนื้อหาที่เรียนรู้จากโปรเจกต์นี้")

with st.expander("🔍 ดูรายละเอียดเนื้อหา"):
    st.markdown("""
    ### 1️⃣ **Supervised Learning**
    - **Classification**: KNN, Decision Tree, SVM
    - **Regression**: Linear Regression
    
    ### 2️⃣ **Unsupervised Learning**
    - **Clustering**: K-Means
    
    ### 3️⃣ **Ensemble Methods**
    - **Bagging**: Random Forest
    - **Boosting**: Gradient Boosting
    - **Voting**: Multi-model ensemble
    
    ### 4️⃣ **ทักษะที่ได้ฝึก**
    - ✅ การทำ Data Preprocessing
    - ✅ Feature Scaling (StandardScaler)
    - ✅ Model Training & Evaluation
    - ✅ Data Visualization
    - ✅ Model Comparison
    - ✅ Hyperparameter Tuning
    """)

# ========== Footer ==========
st.markdown("---")
st.markdown(f"""
<div style="text-align:center; color:#888; padding:20px;">
    <p>🤖 <b>ML Models Explorer</b> | โปรเจกต์วิชา Machine Learning</p>
    <p>จัดทำโดย <b>{STUDENT_NAME}</b> | {STUDENT_SECTION} | {STUDENT_PROGRAM}</p>
    <p>© {datetime.now().year} | Made with ❤️ using Streamlit + scikit-learn</p>
</div>
""", unsafe_allow_html=True)