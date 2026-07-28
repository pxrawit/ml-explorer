import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import os

st.set_page_config(page_title="Decision Tree - Heart Disease", page_icon="🌳", layout="wide")

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
    .risk-high {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
    }
    .risk-low {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🌳 Decision Tree - Heart Disease Prediction")
st.markdown("### ทำนายความเสี่ยงโรคหัวใจด้วยต้นไม้ตัดสินใจ")

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
    st.error("❌ ไม่พบไฟล์ heart_disease_patient_eda_2000_records2.xlsx")
    return None

df = load_data()
if df is None:
    st.stop()

# ========== แสดงข้อมูล ==========
with st.expander("📊 ดูข้อมูลตัวอย่าง", expanded=False):
    st.dataframe(df.head(10))
    st.write(f"**จำนวนข้อมูล:** {len(df):,} records")
    st.write(f"**จำนวน Features:** {len(df.columns) - 1}")
    
    st.markdown("### 📈 สถิติสรุป")
    st.dataframe(df.describe())

# ========== Preprocessing ==========
# เลือก features ที่สำคัญ
numeric_features = ['age', 'bmi', 'systolic_bp', 'diastolic_bp', 
                    'cholesterol_mg_dl', 'fasting_blood_sugar', 
                    'max_heart_rate', 'risk_score']

categorical_features = ['sex', 'smoking_status', 'exercise_level', 
                        'diabetes', 'family_history', 'chest_pain_type', 
                        'ecg_result']

# Encode categorical features
label_encoders = {}
df_encoded = df.copy()

for col in categorical_features:
    if col in df.columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# เลือก features สำหรับ model
feature_cols = numeric_features + [col for col in categorical_features if col in df.columns]
X = df_encoded[feature_cols]
y = df_encoded['heart_disease']

# ========== Train/Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
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
    ["gini", "entropy", "log_loss"],
    help="ฟังก์ชันที่ใช้วัด quality ของ split"
)

# ========== เทรนโมเดล ==========
@st.cache_resource
def train_tree(max_depth, min_samples_split, min_samples_leaf, criterion, X_train, y_train):
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

model = train_tree(max_depth, min_samples_split, min_samples_leaf, criterion, X_train, y_train)

# ========== ประเมินโมเดล ==========
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)

# ========== แสดงผล Metrics ==========
st.markdown("### 📈 ผลลัพธ์การเทรนโมเดล")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy:.4f}")
col2.metric("Train Samples", f"{len(X_train):,}")
col3.metric("Test Samples", f"{len(X_test):,}")
col4.metric("Features", len(feature_cols))

st.info(f"💡 **max_depth={max_depth}** | **criterion={criterion}** | **min_samples_split={min_samples_split}**")

# ========== Confusion Matrix ==========
st.markdown("### 🔢 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
im = ax_cm.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax_cm.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
ax_cm.set_xlabel('Predicted', fontsize=12)
ax_cm.set_ylabel('Actual', fontsize=12)

# Add text annotations
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax_cm.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=16, fontweight='bold')

ax_cm.set_xticks([0, 1])
ax_cm.set_yticks([0, 1])
ax_cm.set_xticklabels(['No Disease (0)', 'Disease (1)'])
ax_cm.set_yticklabels(['No Disease (0)', 'Disease (1)'])
fig_cm.tight_layout()
st.pyplot(fig_cm)

# ========== Classification Report ==========
st.markdown("### 📋 Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

# ========== ROC Curve ==========
st.markdown("### 📉 ROC Curve")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(
    x=fpr, y=tpr,
    mode='lines',
    name=f'ROC Curve (AUC = {roc_auc:.4f})',
    line=dict(color='#56ab2f', width=3)
))
fig_roc.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode='lines',
    name='Random Classifier',
    line=dict(color='red', dash='dash')
))
fig_roc.update_layout(
    title='Receiver Operating Characteristic (ROC) Curve',
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    height=500
)
st.plotly_chart(fig_roc, use_container_width=True)

# ========== Tree Visualization ==========
st.markdown("### 🌲 โครงสร้าง Decision Tree")

tab1, tab2 = st.tabs(["📊 Tree Visualization", "📈 Feature Importance"])

with tab1:
    fig_tree, ax_tree = plt.subplots(figsize=(20, 12))
    plot_tree(
        model,
        feature_names=feature_cols,
        class_names=['No Disease', 'Disease'],
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax_tree,
        impurity=True,
        proportion=False
    )
    ax_tree.set_title(f'Decision Tree (max_depth={max_depth}, criterion={criterion})', 
                      fontsize=16, fontweight='bold')
    st.pyplot(fig_tree)
    
    st.markdown("""
    **วิธีอ่าน Tree:**
    - **กล่องบนสุด (root)**: เงื่อนไขแรกที่แบ่งข้อมูลทั้งหมด
    - **gini/entropy**: ค่า impurity ของ node
    - **samples**: จำนวนตัวอย่างใน node
    - **value**: [จำนวน class 0, จำนวน class 1]
    - **class**: class ที่ทำนายได้ (class ที่มีจำนวนมากกว่า)
    - **สีเข้ม**: มีความแน่นอนสูง
    - **สีอ่อน**: มีความแน่นอนต่ำ
    """)

with tab2:
    # Feature Importance
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
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
    fig_imp.update_layout(height=600)
    st.plotly_chart(fig_imp, use_container_width=True)
    
    st.markdown("""
    **Feature ที่สำคัญที่สุดสำหรับ Heart Disease:**
    - 🫀 **chest_pain_type**: ประเภทอาการเจ็บหน้าอก
    - 💓 **max_heart_rate**: อัตราการเต้นหัวใจสูงสุด
    - 🩸 **systolic_bp**: ความดันโลหิตตัวบน
    - 📊 **risk_score**: คะแนนความเสี่ยง
    - 🧬 **age**: อายุ
    """)

# ========== Decision Boundary ==========
st.markdown("### 🗺️ Decision Boundary (2D Visualization)")

st.markdown("เลือก 2 features เพื่อดูว่า Decision Tree แบ่งพื้นที่อย่างไร")

col_a, col_b = st.columns(2)
with col_a:
    feature_x = st.selectbox("Feature แกน X", feature_cols, index=feature_cols.index('age'))
with col_b:
    feature_y = st.selectbox("Feature แกน Y", feature_cols, index=feature_cols.index('max_heart_rate'))

# สร้าง model ใหม่ด้วย 2 features
X_2d = X[[feature_x, feature_y]]
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y, test_size=0.2, random_state=42, stratify=y
)

model_2d = DecisionTreeClassifier(
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    criterion=criterion,
    random_state=42
)
model_2d.fit(X_train_2d, y_train_2d)

# สร้าง mesh grid
x_min, x_max = X_2d[feature_x].min() - 1, X_2d[feature_x].max() + 1
y_min, y_max = X_2d[feature_y].min() - 1, X_2d[feature_y].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 100),
    np.linspace(y_min, y_max, 100)
)
grid = np.c_[xx.ravel(), yy.ravel()]
Z = model_2d.predict(grid).reshape(xx.shape)

fig_boundary = go.Figure()

# Decision boundary
fig_boundary.add_trace(go.Contour(
    x=np.linspace(x_min, x_max, 100),
    y=np.linspace(y_min, y_max, 100),
    z=Z,
    colorscale=['#a8e063', '#ff416c'],
    opacity=0.5,
    showscale=False,
    name='Decision Boundary'
))

# Data points
fig_boundary.add_trace(go.Scatter(
    x=X_test_2d[feature_x],
    y=X_test_2d[feature_y],
    mode='markers',
    marker=dict(
        color=y_test_2d,
        colorscale=['#56ab2f', '#ff416c'],
        size=8,
        line=dict(width=1, color='black')
    ),
    name='Test Data',
    text=[f'Actual: {int(val)}' for val in y_test_2d]
))

fig_boundary.update_layout(
    xaxis_title=feature_x,
    yaxis_title=feature_y,
    height=600,
    title=f'Decision Boundary: {feature_x} vs {feature_y}'
)
st.plotly_chart(fig_boundary, use_container_width=True)

# ========== ส่วนทำนาย ==========
st.markdown("---")
st.markdown("## 🎯 ลองทำนายความเสี่ยงโรคหัวใจ!")

st.sidebar.markdown("---")
st.sidebar.header("🏥 กรอกข้อมูลผู้ป่วย")

# สร้าง input สำหรับแต่ละ feature
input_data = {}

# Numeric features
for feature in numeric_features:
    if feature in df.columns:
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        default_val = float(df[feature].median())
        
        if feature in ['age', 'bmi', 'risk_score']:
            input_data[feature] = st.sidebar.slider(
                f"📊 {feature}",
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                step=0.1
            )
        elif feature in ['systolic_bp', 'diastolic_bp', 'cholesterol_mg_dl', 
                         'fasting_blood_sugar', 'max_heart_rate']:
            input_data[feature] = st.sidebar.number_input(
                f"📊 {feature}",
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                step=1.0
            )

# Categorical features
for feature in categorical_features:
    if feature in df.columns:
        unique_vals = df[feature].unique()
        default_val = df[feature].mode()[0]
        
        if feature == 'sex':
            input_data[feature] = st.sidebar.selectbox(
                f"👤 {feature}",
                options=unique_vals,
                index=list(unique_vals).index(default_val)
            )
        elif feature in ['diabetes', 'family_history']:
            input_data[feature] = st.sidebar.selectbox(
                f"🧬 {feature}",
                options=unique_vals,
                index=list(unique_vals).index(default_val)
            )
        elif feature in ['smoking_status', 'exercise_level', 'chest_pain_type', 'ecg_result']:
            input_data[feature] = st.sidebar.selectbox(
                f"🏥 {feature}",
                options=unique_vals,
                index=list(unique_vals).index(default_val)
            )

# Encode categorical features for prediction
input_encoded = {}
for col in feature_cols:
    if col in numeric_features:
        input_encoded[col] = input_data[col]
    elif col in categorical_features and col in label_encoders:
        try:
            input_encoded[col] = label_encoders[col].transform([str(input_data[col])])[0]
        except:
            input_encoded[col] = 0

# ========== ทำนาย ==========
input_df = pd.DataFrame([input_encoded])
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0]

# ========== แสดงผลทำนาย ==========
st.markdown("### 🎉 ผลทำนาย")

if prediction == 1:
    risk_level = "มีความเสี่ยงสูง"
    risk_class = "risk-high"
    emoji = "⚠️"
else:
    risk_level = "ความเสี่ยงต่ำ"
    risk_class = "risk-low"
    emoji = "✅"

st.markdown(f"""
<div class="prediction-box {risk_class}">
    <div style="font-size:1.5em;">{emoji} {risk_level}</div>
    <div class="prediction-value">{'เป็นโรคหัวใจ' if prediction == 1 else 'ไม่เป็นโรคหัวใจ'}</div>
    <div style="font-size:1.2em;">ความน่าจะเป็น: {prediction_proba[prediction]*100:.2f}%</div>
</div>
""", unsafe_allow_html=True)

# แสดงความน่าจะเป็นของแต่ละ class
st.markdown("### 📊 ความน่าจะเป็นของแต่ละ Class")

fig_proba = go.Figure()
fig_proba.add_trace(go.Bar(
    x=['ไม่เป็นโรค (0)', 'เป็นโรค (1)'],
    y=prediction_proba,
    marker_color=['#56ab2f', '#ff416c'],
    text=[f'{p*100:.2f}%' for p in prediction_proba],
    textposition='outside'
))
fig_proba.update_layout(
    title='Probability Distribution',
    yaxis_title='Probability',
    yaxis_range=[0, 1],
    height=400
)
st.plotly_chart(fig_proba, use_container_width=True)

# ========== Decision Path ==========
st.markdown("### 🛤️ Decision Path (เส้นทางที่ต้นไม้ตัดสินใจ)")

node_indicator = model.decision_path(input_df)
feature_arr = model.tree_.feature
threshold_arr = model.tree_.threshold

path_info = []
node_index = node_indicator.indices[node_indicator.nonzero()]

for node_id in node_index:
    if node_id == node_index[-1]:  # leaf node
        path_info.append({
            'Node': node_id,
            'Condition': f'🎯 LEAF NODE → Predict: {"Disease" if model.tree_.value[node_id][0][1] > model.tree_.value[node_id][0][0] else "No Disease"}',
            'Samples': model.tree_.n_node_samples[node_id],
            'Value': f"[{int(model.tree_.value[node_id][0][0])}, {int(model.tree_.value[node_id][0][1])}]"
        })
    else:
        feature_name = feature_cols[feature_arr[node_id]]
        threshold = threshold_arr[node_id]
        value = input_df[feature_name].values[0]
        
        if value <= threshold:
            path_info.append({
                'Node': node_id,
                'Condition': f'✅ {feature_name} ≤ {threshold:.2f} (ค่าจริง: {value:.2f})',
                'Samples': model.tree_.n_node_samples[node_id],
                'Value': f"[{int(model.tree_.value[node_id][0][0])}, {int(model.tree_.value[node_id][0][1])}]"
            })
        else:
            path_info.append({
                'Node': node_id,
                'Condition': f'❌ {feature_name} > {threshold:.2f} (ค่าจริง: {value:.2f})',
                'Samples': model.tree_.n_node_samples[node_id],
                'Value': f"[{int(model.tree_.value[node_id][0][0])}, {int(model.tree_.value[node_id][0][1])}]"
            })

path_df = pd.DataFrame(path_info)
st.dataframe(path_df, use_container_width=True, hide_index=True)

# ========== Distribution Analysis ==========
st.markdown("### 📊 การกระจายตัวของข้อมูลตามกลุ่ม")

tab3, tab4 = st.tabs(["📈 Numeric Features", "📊 Categorical Features"])

with tab3:
    selected_feature = st.selectbox(
        "เลือก Feature เพื่อดูการกระจายตัว",
        numeric_features,
        index=0
    )
    
    fig_dist = px.histogram(
        df,
        x=selected_feature,
        color='heart_disease',
        marginal='box',
        title=f'การกระจายตัวของ {selected_feature} ตามกลุ่มโรคหัวใจ',
        color_discrete_map={0: '#56ab2f', 1: '#ff416c'},
        labels={'heart_disease': 'โรคหัวใจ'}
    )
    fig_dist.update_layout(height=500)
    st.plotly_chart(fig_dist, use_container_width=True)

with tab4:
    selected_cat = st.selectbox(
        "เลือก Categorical Feature",
        categorical_features,
        index=0
    )
    
    fig_cat = px.histogram(
        df,
        x=selected_cat,
        color='heart_disease',
        barmode='group',
        title=f'การกระจายตัวของ {selected_cat} ตามกลุ่มโรคหัวใจ',
        color_discrete_map={0: '#56ab2f', 1: '#ff416c'},
        labels={'heart_disease': 'โรคหัวใจ'}
    )
    fig_cat.update_layout(height=500)
    st.plotly_chart(fig_cat, use_container_width=True)

# ========== ข้อดี/ข้อเสีย ==========
st.markdown("---")
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
    - **เหมาะสำหรับข้อมูลทางการแพทย์** - อธิบายให้หมอเข้าใจได้
    """)

with col2:
    st.error("""
    **❌ ข้อเสีย:**
    - **Overfit ได้ง่ายมาก** - ต้อง prune หรือจำกัด depth
    - **ไม่ stable** - ข้อมูลเปลี่ยนนิดเดียว ต้นไม้เปลี่ยนเยอะ
    - **Greedy algorithm** - ไม่การันตี global optimum
    - **จับความสัมพันธ์เชิงเส้น** ไม่ดีเท่า linear models
    - **Bias ต่อ feature ที่มีค่ามาก**
    - **Extrapolate ไม่ได้** - ทำนายค่าที่อยู่นอกช่วง training ไม่ได้
    - **ควรใช้ Random Forest** แทนสำหรับ production
    """)

# ========== Tips ==========
st.markdown("### 💡 เคล็ดลับการใช้งาน Decision Tree สำหรับ Heart Disease")

st.info("""
**1. ป้องกัน Overfitting:**
- เริ่มจาก `max_depth=3-5` แล้วค่อยๆ เพิ่ม
- ใช้ `min_samples_leaf=5-10` เพื่อลด leaf ที่มีตัวอย่างน้อย
- ลองใช้ **Random Forest** (ensemble ของ trees) เพื่อลด variance

**2. Feature Engineering:**
- Decision Tree จับ interaction ระหว่าง features ได้ดี
- ลองสร้าง features ใหม่เช่น `bp_ratio = systolic_bp / diastolic_bp`
- ใช้ `risk_score` ที่มีอยู่แล้วเป็น feature สำคัญ

**3. Interpretation:**
- ใช้ tree visualization เพื่ออธิบายให้หมอเข้าใจ
- ดู decision path ว่าใช้เงื่อนไขอะไรบ้าง
- Feature importance ช่วยระบุว่าอะไรสำคัญที่สุด

**4. Production:**
- ใช้ Random Forest หรือ Gradient Boosting แทน
- Validate ด้วย cross-validation
- Monitor performance อย่างสม่ำเสมอ
""")

st.markdown("---")
st.caption("🌳 Decision Tree Heart Disease Predictor | Machine Learning Explorer")