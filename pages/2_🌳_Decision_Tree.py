import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

st.set_page_config(page_title="Decision Tree - California Housing", page_icon="🌳", layout="wide")

# ========== Custom CSS ==========
st.markdown("""
<style>
    .prediction-box {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: #000;
        margin: 20px 0;
    }
    .prediction-value {
        font-size: 3em;
        font-weight: bold;
        margin: 10px 0;
    }
    .step-box {
        background: rgba(86, 171, 47, 0.1);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 3px solid #56ab2f;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🌳 Decision Tree - California Housing Price Prediction")
st.markdown("### ทำนายราคาบ้านด้วยต้นไม้ตัดสินใจ - แบ่งข้อมูลด้วยกฎ if-else")

# ========== ทฤษฎี ==========
with st.expander("📚 ทฤษฎี Decision Tree Regression", expanded=False):
    st.markdown("""
    **Decision Tree Regression** แบ่งข้อมูลออกเป็นกิ่งๆ ด้วยคำถาม "ใช่/ไม่ใช่" 
    จนกว่าจะได้กลุ่มที่มีค่า target คล้ายกัน
    
    ### 🧮 หลักการทำงาน
    1. หา feature และ threshold ที่ดีที่สุดในการแบ่งข้อมูล
    2. แบ่งข้อมูลเป็น 2 กิ่ง (ซ้าย/ขวา)
    3. ทำซ้ำแบบ recursive จนกว่าจะถึงเงื่อนไขหยุด
    4. ทำนาย = ค่าเฉลี่ยของ leaf node
    
    ### 📏 Criterion สำหรับ Regression
    - **MSE (Mean Squared Error)**: ลดผลรวมของกำลังสองของ error
    - **MAE (Mean Absolute Error)**: ลดผลรวมของค่าสัมบูรณ์ของ error  
    - **Friedman MSE**: ปรับปรุงจาก MSE โดย Friedman
    - **Poisson**: สำหรับข้อมูล count
    
    ### ✂️ การป้องกัน Overfitting
    - `max_depth`: จำกัดความลึกของต้นไม้
    - `min_samples_split`: จำนวนตัวอย่างขั้นต่ำที่ต้องมีเพื่อแบ่ง node
    - `min_samples_leaf`: จำนวนตัวอย่างขั้นต่ำใน leaf
    - `max_features`: จำนวน feature ที่พิจารณาในแต่ละ split
    """)

# ========== โหลดข้อมูล ==========
@st.cache_data
def load_data():
    paths = [
        "california_housing_test.csv",
        "data/california_housing_test.csv",
        "../california_housing_test.csv"
    ]
    for path in paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing(as_frame=True)
    return data.frame.reset_index(drop=True)

try:
    df = load_data()
except Exception as e:
    st.error(f"❌ โหลดข้อมูลไม่สำเร็จ: {e}")
    st.stop()

# ========== แยก Features/Target ==========
X = df.drop(columns=['median_house_value'])
y = df['median_house_value']
feature_names = X.columns.tolist()

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========== Sidebar: ปรับพารามิเตอร์ ==========
st.sidebar.header("⚙️ พารามิเตอร์ Decision Tree")

max_depth = st.sidebar.slider(
    "ความลึกสูงสุด (max_depth)", 
    min_value=1, 
    max_value=20, 
    value=5,
    help="จำกัดความลึกของต้นไม้ - มาก = overfit, น้อย = underfit"
)

min_samples_split = st.sidebar.slider(
    "ตัวอย่างขั้นต่ำเพื่อแบ่ง (min_samples_split)",
    min_value=2,
    max_value=50,
    value=10,
    help="จำนวนตัวอย่างขั้นต่ำที่ต้องมีใน node เพื่อจะแบ่งต่อ"
)

min_samples_leaf = st.sidebar.slider(
    "ตัวอย่างขั้นต่ำใน leaf (min_samples_leaf)",
    min_value=1,
    max_value=20,
    value=5,
    help="จำนวนตัวอย่างขั้นต่ำใน leaf node"
)

criterion = st.sidebar.selectbox(
    "Criterion (เกณฑ์การแบ่ง)",
    ["squared_error", "absolute_error", "friedman_mse", "poisson"],
    help="ฟังก์ชันที่ใช้วัด quality ของ split"
)

# ========== เทรนโมเดล ==========
@st.cache_resource
def train_tree(max_depth, min_samples_split, min_samples_leaf, criterion, X_train, y_train):
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

with st.spinner("กำลังเทรน Decision Tree..."):
    model = train_tree(max_depth, min_samples_split, min_samples_leaf, criterion, X_train, y_train)

# ========== ประเมินโมเดล ==========
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ========== แสดงผล Metrics ==========
st.markdown("### 📈 ผลลัพธ์การเทรนโมเดล")
col1, col2, col3, col4 = st.columns(4)
col1.metric("R² Score", f"{r2:.4f}", help="ยิ่งใกล้ 1 ยิ่งดี")
col2.metric("RMSE", f"${rmse:,.0f}", help="Root Mean Squared Error")
col3.metric("MAE", f"${mae:,.0f}", help="Mean Absolute Error")
col4.metric("MSE", f"{mse:,.0f}", help="Mean Squared Error")

st.info(f"💡 **max_depth={max_depth}** | **min_samples_split={min_samples_split}** | **min_samples_leaf={min_samples_leaf}** | **criterion={criterion}**")

# ========== Tree Visualization ==========
st.markdown("### 🌲 โครงสร้าง Decision Tree")

tab1, tab2 = st.tabs(["📊 Tree Visualization", "📈 Feature Importance"])

with tab1:
    fig_tree, ax_tree = plt.subplots(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax_tree,
        impurity=False,
        proportion=True
    )
    ax_tree.set_title(f'Decision Tree (max_depth={max_depth}, criterion={criterion})', fontsize=16)
    st.pyplot(fig_tree)
    
    st.markdown("""
    **วิธีอ่าน Tree:**
    - **กล่องบนสุด (root)**: เงื่อนไขแรกที่แบ่งข้อมูลทั้งหมด
    - **ตัวเลขในกล่อง**: ค่าเฉลี่ยของ target ใน node นั้น
    - **sample**: จำนวนตัวอย่างใน node
    - **สีเข้ม**: ค่า target สูง (บ้านแพง)
    - **สีอ่อน**: ค่า target ต่ำ (บ้านถูก)
    """)

with tab2:
    # Feature Importance
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    fig_imp = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance - Feature ไหนสำคัญที่สุด?',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    fig_imp.update_layout(height=500)
    st.plotly_chart(fig_imp, use_container_width=True)
    
    st.markdown("""
    **Feature ที่สำคัญที่สุดสำหรับ California Housing:**
    - 🏠 **Median Income**: รายได้เฉลี่ยมีผลมากที่สุดต่อราคาบ้าน
    - 📍 **Latitude/Longitude**: ตำแหน่งที่ตั้งสำคัญมาก
    - 🏘️ **Ave Occup**: จำนวนคนต่อครัวเรือน
    - 🏚️ **Housing Age**: อายุบ้านมีผลปานกลาง
    """)

# ========== Predicted vs Actual ==========
st.markdown("### 🔍 Predicted vs Actual Values")
fig_scatter = go.Figure()
fig_scatter.add_trace(go.Scatter(
    x=y_test, y=y_pred,
    mode='markers',
    marker=dict(color='#56ab2f', size=5, opacity=0.5),
    name='Predictions'
))
max_val = max(y_test.max(), y_pred.max())
fig_scatter.add_trace(go.Scatter(
    x=[0, max_val], y=[0, max_val],
    mode='lines',
    line=dict(color='red', dash='dash', width=2),
    name='Perfect Prediction'
))
fig_scatter.update_layout(
    xaxis_title="Actual Price ($)",
    yaxis_title="Predicted Price ($)",
    height=400
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ========== Decision Boundary (ใช้ 2 features) ==========
st.markdown("### 🗺️ Decision Boundary (Latitude vs Longitude)")
st.markdown("ดูว่า Decision Tree แบ่งพื้นที่ California ออกเป็น region ต่างๆ อย่างไร")

col_a, col_b = st.columns(2)
with col_a:
    feature_x = st.selectbox("Feature แกน X", feature_names, index=feature_names.index('longitude'))
with col_b:
    feature_y = st.selectbox("Feature แกน Y", feature_names, index=feature_names.index('latitude'))

# สร้าง grid
x_min, x_max = X[feature_x].min() - 0.1, X[feature_x].max() + 0.1
y_min, y_max = X[feature_y].min() - 0.1, X[feature_y].max() + 0.1

# สร้าง dataframe ใหม่ที่มีแค่ 2 features สำหรับ visualization
X_2d = X[[feature_x, feature_y]]
X_train_2d, _, _, _ = train_test_split(X_2d, y, test_size=0.2, random_state=42)

model_2d = DecisionTreeRegressor(
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    criterion=criterion,
    random_state=42
)
model_2d.fit(X_train_2d, y_train)

# สร้าง mesh grid
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 100),
    np.linspace(y_min, y_max, 100)
)
grid = np.c_[xx.ravel(), yy.ravel()]
Z = model_2d.predict(grid).reshape(xx.shape)

fig_boundary = go.Figure()
fig_boundary.add_trace(go.Contour(
    x=np.linspace(x_min, x_max, 100),
    y=np.linspace(y_min, y_max, 100),
    z=Z,
    colorscale='Viridis',
    opacity=0.7,
    showscale=True,
    colorbar=dict(title="Price ($)")
))
fig_boundary.add_trace(go.Scatter(
    x=X[feature_x].sample(500, random_state=42),
    y=X[feature_y].sample(500, random_state=42),
    mode='markers',
    marker=dict(color='white', size=3, opacity=0.3),
    name='Data Points'
))
fig_boundary.update_layout(
    xaxis_title=feature_x,
    yaxis_title=feature_y,
    height=500,
    title=f'Decision Boundary: {feature_x} vs {feature_y}'
)
st.plotly_chart(fig_boundary, use_container_width=True)

# ========== ส่วนทำนาย ==========
st.markdown("---")
st.markdown("## 🎯 ลองทำนายราคาบ้านของคุณเอง!")
st.markdown("ปรับค่า features ด้านล่างแล้วดูการทำนาย")

st.sidebar.markdown("---")
st.sidebar.header("🏠 กรอกข้อมูลบ้าน")

feature_ranges = {
    'longitude': (-124.35, -114.31),
    'latitude': (32.54, 41.95),
    'housing_median_age': (1.0, 52.0),
    'total_rooms': (6.0, 39320.0),
    'total_bedrooms': (1.0, 6445.0),
    'population': (3.0, 35682.0),
    'households': (1.0, 6082.0),
    'median_income': (0.50, 15.00)
}

feature_labels = {
    'longitude': '🌍 Longitude (ลองจิจูด)',
    'latitude': '🌎 Latitude (ละติจูด)',
    'housing_median_age': '🏚️ อายุบ้านเฉลี่ย (ปี)',
    'total_rooms': '🚪 จำนวนห้องทั้งหมด',
    'total_bedrooms': '🛏️ จำนวนห้องนอน',
    'population': '👥 จำนวนประชากร',
    'households': '🏘️ จำนวนครัวเรือน',
    'median_income': '💰 รายได้เฉลี่ย (x$10,000)'
}

input_values = {}
for feature in feature_names:
    min_val, max_val = feature_ranges.get(feature, (0, 100))
    default_val = float(X[feature].median())
    
    if feature in ['longitude', 'latitude']:
        input_values[feature] = st.sidebar.slider(
            feature_labels[feature],
            min_value=float(min_val),
            max_value=float(max_val),
            value=default_val,
            step=0.01
        )
    elif feature == 'median_income':
        input_values[feature] = st.sidebar.slider(
            feature_labels[feature],
            min_value=float(min_val),
            max_value=float(max_val),
            value=default_val,
            step=0.1
        )
    else:
        input_values[feature] = st.sidebar.number_input(
            feature_labels[feature],
            min_value=float(min_val),
            max_value=float(max_val),
            value=default_val,
            step=1.0
        )

# ========== ทำนาย ==========
input_df = pd.DataFrame([input_values])
prediction = model.predict(input_df)[0]

# ========== แสดงผลทำนาย ==========
st.markdown("### 🎉 ผลทำนาย")

st.markdown(f"""
<div class="prediction-box">
    <div style="font-size:1.2em;">💵 ราคาบ้านที่ทำนายได้</div>
    <div class="prediction-value">${prediction:,.0f}</div>
    <div style="font-size:1.1em;">หรือประมาณ {prediction*36:,.0f} บาท</div>
</div>
""", unsafe_allow_html=True)

# ========== แสดง Decision Path ==========
st.markdown("### 🛤️ Decision Path (เส้นทางที่ต้นไม้ตัดสินใจ)")
st.markdown("ดูว่า Decision Tree ใช้เงื่อนไขอะไรบ้างเพื่อทำนายบ้านหลังนี้")

node_indicator = model.decision_path(input_df)
feature_arr = model.tree_.feature
threshold_arr = model.tree_.threshold

path_info = []
node_index = node_indicator.indices[node_indicator.nonzero()]

for node_id in node_index:
    if node_id == node_index[-1]:  # leaf node
        path_info.append({
            'Node': node_id,
            'Condition': f'🎯 LEAF NODE → Predict: ${model.tree_.value[node_id][0][0]:,.0f}',
            'Samples': model.tree_.n_node_samples[node_id]
        })
    else:
        feature_name = feature_names[feature_arr[node_id]]
        threshold = threshold_arr[node_id]
        value = input_df[feature_name].values[0]
        
        if value <= threshold:
            path_info.append({
                'Node': node_id,
                'Condition': f'✅ {feature_name} ≤ {threshold:.2f} (ค่าจริง: {value:.2f})',
                'Samples': model.tree_.n_node_samples[node_id]
            })
        else:
            path_info.append({
                'Node': node_id,
                'Condition': f'❌ {feature_name} > {threshold:.2f} (ค่าจริง: {value:.2f})',
                'Samples': model.tree_.n_node_samples[node_id]
            })

path_df = pd.DataFrame(path_info)
st.dataframe(path_df, use_container_width=True, hide_index=True)

# ========== Feature Distribution ==========
st.markdown("### 📊 การกระจายตัวของ Features ที่คุณกรอก")

col_a, col_b = st.columns(2)
with col_a:
    fig_inc = px.histogram(
        X, x='median_income', nbins=50,
        title='การกระจายตัวของรายได้เฉลี่ย',
        labels={'median_income': 'Median Income (x$10k)'}
    )
    fig_inc.add_vline(x=input_values['median_income'], line_dash="dash", line_color="red",
                      annotation_text=f"คุณ: {input_values['median_income']:.2f}")
    st.plotly_chart(fig_inc, use_container_width=True)

with col_b:
    fig_age = px.histogram(
        X, x='housing_median_age', nbins=50,
        title='การกระจายตัวของอายุบ้าน',
        labels={'housing_median_age': 'Housing Median Age'}
    )
    fig_age.add_vline(x=input_values['housing_median_age'], line_dash="dash", line_color="red",
                      annotation_text=f"คุณ: {input_values['housing_median_age']:.0f}")
    st.plotly_chart(fig_age, use_container_width=True)

# ========== เปรียบเทียบกับ KNN ==========
st.markdown("---")
st.markdown("### ⚖️ เปรียบเทียบ Decision Tree vs KNN")

# Train KNN สำหรับเปรียบเทียบ
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

comparison_df = pd.DataFrame({
    'Model': ['Decision Tree', 'KNN'],
    'R² Score': [r2, r2_score(y_test, y_pred_knn)],
    'RMSE ($)': [rmse, np.sqrt(mean_squared_error(y_test, y_pred_knn))],
    'MAE ($)': [mae, mean_absolute_error(y_test, y_pred_knn)],
    'Training Time': ['เร็วมาก ⚡', 'ช้ากว่า (ต้องคำนวณระยะทาง) 🐢'],
    'Interpretability': ['สูงมาก (เห็นกฎชัดเจน) ✅', 'ต่ำ (black box) ❌'],
    'Overfitting Risk': ['สูง (ต้อง prune) ⚠️', 'ต่ำ (ขึ้นกับ K) ✅']
})

st.dataframe(comparison_df, use_container_width=True, hide_index=True)

fig_comp = go.Figure()
fig_comp.add_trace(go.Bar(
    x=comparison_df['Model'],
    y=comparison_df['R² Score'],
    name='R² Score',
    marker_color=['#56ab2f', '#00c9ff']
))
fig_comp.update_layout(
    title='เปรียบเทียบ R² Score: Decision Tree vs KNN',
    yaxis_title='R² Score',
    height=400
)
st.plotly_chart(fig_comp, use_container_width=True)

# ========== ข้อดี/ข้อเสีย ==========
st.markdown("### ✅ ข้อดี / ❌ ข้อเสียของ Decision Tree")

col1, col2 = st.columns(2)
with col1:
    st.success("""
    **✅ ข้อดี:**
    - **ตีความง่าย** - เห็นกฎการตัดสินใจชัดเจน
    - **ไม่ต้อง scale ข้อมูล** - ทำงานกับข้อมูลดิบได้เลย
    - **จัดการทั้ง numeric และ categorical** ได้
    - **เร็วทั้ง train และ predict**
    - **เห็น feature importance** ชัดเจน
    - **ไม่สมมติการกระจายตัว** ของข้อมูล
    """)

with col2:
    st.error("""
    **❌ ข้อเสีย:**
    - **Overfit ได้ง่ายมาก** - ต้อง prune หรือจำกัด depth
    - **ไม่ stable** - ข้อมูลเปลี่ยนนิดเดียว ต้นไม้เปลี่ยนเยอะ
    - **Greedy algorithm** - ไม่การันตี global optimum
    - **จับความสัมพันธ์เชิงเส้น** ไม่ดีเท่า linear models
    - **Bias ต่อ feature ที่มีค่ามาก** (เช่น total_rooms)
    - **Extrapolate ไม่ได้** - ทำนายค่าที่อยู่นอกช่วง training ไม่ได้
    """)

# ========== Tips ==========
st.markdown("### 💡 เคล็ดลับการใช้งาน Decision Tree")

st.info("""
**1. ป้องกัน Overfitting:**
- เริ่มจาก `max_depth=3-5` แล้วค่อยๆ เพิ่ม
- ใช้ `min_samples_leaf=5-10` เพื่อลด leaf ที่มีตัวอย่างน้อย
- ลองใช้ **Random Forest** (ensemble ของ trees) เพื่อลด variance

**2. เลือก Criterion:**
- `squared_error`: ค่า default, ทำงานดีกับข้อมูลส่วนใหญ่
- `absolute_error`: ทนทานต่อ outlier มากกว่า
- `friedman_mse`: ปรับปรุงสำหรับ regression

**3. Feature Engineering:**
- Decision Tree จับ interaction ระหว่าง features ได้ดี
- ลองสร้าง features ใหม่เช่น `rooms_per_household = total_rooms / households`

**4. เปรียบเทียบกับโมเดลอื่น:**
- ถ้าข้อมูลมีความสัมพันธ์เชิงเส้น → ลอง **Linear Regression**
- ถ้าต้องการความแม่นยำสูง → ลอง **Random Forest** หรือ **Gradient Boosting**
- ถ้าต้องการ interpretability → **Decision Tree** คือคำตอบ
""")

st.markdown("---")
st.caption("🌳 Decision Tree California Housing Predictor | Machine Learning Explorer")