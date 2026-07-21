import streamlit as st

st.set_page_config(
    page_title="ML Models Explorer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    .model-card {
        background: linear-gradient(135deg, rgba(0,201,255,0.1), rgba(146,254,157,0.1));
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
        transition: 0.3s;
    }
    .model-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,201,255,0.3);
    }
    .icon { font-size: 3em; margin-bottom: 10px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🤖 ML Models Explorer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">สำรวจ 6 อัลกอริทึม Machine Learning พื้นฐาน พร้อม Interactive Demo</p>', unsafe_allow_html=True)

st.markdown("---")

st.markdown("### 👈 เลือกโมเดลจากเมนูด้านซ้ายเพื่อเริ่มเรียนรู้")

st.markdown("### 📚 โมเดลทั้งหมด")

cols = st.columns(3)
models = [
    ("📍", "KNN", "K-Nearest Neighbors\nทำนายจากเพื่อนบ้านที่ใกล้ที่สุด"),
    ("🌳", "Decision Tree", "ต้นไม้ตัดสินใจ\nแบ่งข้อมูลด้วยกฎ if-else"),
    ("📈", "Regression", "Linear Regression\nหาเส้นตรงที่ fit ข้อมูลดีที่สุด"),
    ("⚔️", "SVM", "Support Vector Machine\nหาเส้นแบ่งที่กว้างที่สุด"),
    ("🎯", "K-Means", "Clustering\nจัดกลุ่มข้อมูลเป็น K คลัสเตอร์"),
    ("👥", "Ensemble", "Random Forest / Voting\nรวมหลายโมเดลเพื่อทำนายแม่นยำขึ้น"),
]

for i, (icon, name, desc) in enumerate(models):
    with cols[i % 3]:
        st.markdown(f"""
        <div class="model-card">
            <div class="icon">{icon}</div>
            <h3>{name}</h3>
            <p style="color:#bbb; font-size:0.9em;">{desc.replace(chr(10), '<br>')}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("### 🎯 วิธีใช้งาน")
st.info("""
1. **เลือกโมเดล** จากเมนูด้านซ้ายมือ
2. **ปรับพารามิเตอร์** ใน sidebar เพื่อทดลอง
3. **ดูผลลัพธ์** จาก visualization และคำอธิบาย
4. **เรียนรู้** ข้อดี-ข้อเสียของแต่ละโมเดล
""")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit + scikit-learn | © 2026")