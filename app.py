import streamlit as st

st.set_page_config(page_title="ML Models Explorer", page_icon="🤖", layout="centered")

st.title("🤖 ML Models Explorer")
st.write("เลือกโมเดลที่ต้องการดูจากเมนูด้านล่าง หรือแถบด้านซ้าย")

st.page_link("pages/1_📍_KNN.py", label="1. KNN", icon="📍")
st.page_link("pages/2_🌳_Decision_Tree.py", label="2. Decision Tree", icon="🌳")
st.page_link("pages/3_📈_Regression.py", label="3. Regression", icon="📈")
st.page_link("pages/4_⚔️_SVM.py", label="4. SVM", icon="⚔️")
st.page_link("pages/5_🎯_K_Means.py", label="5. K-Means", icon="🎯")
st.page_link("pages/6_👥_Ensemble.py", label="6. Ensemble", icon="👥")