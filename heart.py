import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time


st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #111111, #000000);
    color: white;
}
h1,h2,h3 { color:#ff4b4b; text-align:center; }
label { color:white !important; }

.stButton > button {
    background: linear-gradient(90deg,#ff4b4b,#ff0066);
    color:white;
    border-radius:30px;
    height:55px;
    font-size:18px;
    width:100%;
}
.stButton > button:hover {
    transform: scale(1.05);
    transition: transform 0.2s;
}

div[data-testid="stNumberInput"]:hover,
div[data-testid="stSelectbox"]:hover {
    transform: scale(1.02);
    transition: transform 0.2s ease-in-out;
    box-shadow: 0px 0px 15px #ff4b4b;
}

div[data-testid="stNumberInput"],
div[data-testid="stSelectbox"] {
    background-color:#1a1a1a;
    color:white;
    padding:10px;
    border-radius:12px;
}
div[data-baseweb="input"] input {
    color:white !important;
    background-color:#1a1a1a !important;
}
div[data-baseweb="select"] span {
    color:white !important;
}


@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}
</style>
""", unsafe_allow_html=True)

model = joblib.load("best_rf_model.pkl")

st.markdown("<h1>Heart Disease Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-powered Clinical Risk Assessment</p>", unsafe_allow_html=True)
st.divider()

st.subheader("Patient Details")
c1, c2 = st.columns(2)

with c1:
    age = st.number_input("Age", 1, 120, 45)
    trestbps = st.number_input("Resting BP", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 400, 200)
    thalch = st.number_input("Max Heart Rate", 60, 220, 150)
    oldpeak = st.number_input("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    ca = st.selectbox("Major Vessels (0-3)", [0,1,2,3])

with c2:
    sex = st.selectbox("Sex", ["Male","Female"])
    cp = st.selectbox("Chest Pain Type", ["typical angina","atypical angina","non-anginal","asymptomatic"])
    fbs = st.selectbox("Fasting Blood Sugar >120", ["True","False"])
    restecg = st.selectbox("Rest ECG", ["normal","lv hypertrophy","st-t abnormality"])
    exang = st.selectbox("Exercise Induced Angina", ["True","False"])
    slope = st.selectbox("ST Slope", ["upsloping","flat","downsloping"])
    thal = st.selectbox("Thalassemia", ["normal","fixed defect","reversable defect"])

sex = 1 if sex=="Male" else 0
fbs = 1 if fbs=="True" else 0
exang = 1 if exang=="True" else 0

cp_map = {"typical angina":0,"atypical angina":1,"non-anginal":2,"asymptomatic":3}
restecg_map = {"normal":0,"lv hypertrophy":1,"st-t abnormality":2}
slope_map = {"upsloping":0,"flat":1,"downsloping":2}
thal_map = {"normal":1,"fixed defect":2,"reversable defect":3}

input_df = pd.DataFrame([{
    "id": 0,
    "age": age,
    "trestbps": trestbps,
    "chol": chol,
    "thalch": thalch,
    "oldpeak": oldpeak,
    "ca": ca,
    "sex": sex,
    "cp": cp_map[cp],
    "fbs": fbs,
    "restecg": restecg_map[restecg],
    "exang": exang,
    "slope": slope_map[slope],
    "thal": thal_map[thal]
}])

st.divider()

if st.button("Predict Heart Disease"):

    with st.spinner(" Analyzing patient data..."):
        time.sleep(1.5)
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]*100

    
    if pred == 1:
        st.markdown(f"""
        <h2 style='color:red; text-align:center; animation:pulse 1s infinite;'> High Risk ({prob:.2f}%)</h2>
        """, unsafe_allow_html=True)
    else:
        st.success(f" Low Risk ({prob:.2f}%)")

        gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,  # directly show final value
        title={'text': "Risk %"},
        gauge={
            'axis': {'range':[0,100]},
            'bar': {'color':'red'},
            'steps':[{'range':[0,40],'color':'green'},
                     {'range':[40,70],'color':'orange'},
                     {'range':[70,100],'color':'red'}]
        }
    ))
    st.plotly_chart(gauge, use_container_width=True, key="gauge_final")

    # ================= PATIENT METRICS CHART =================
    st.subheader("📊 Patient Health Metrics")
    metrics_df = pd.DataFrame({
        "Metric":["Age","BP","Cholesterol","Heart Rate","Oldpeak"],
        "Value":[age,trestbps,chol,thalch,oldpeak]
    })
    fig_metrics = px.bar(
        metrics_df,
        x="Metric",
        y="Value",
        color="Metric",
        template="plotly_dark",
        text="Value",
        range_y=[0, max(metrics_df["Value"])*1.2],
    )
    fig_metrics.update_traces(textposition='outside', hovertemplate='%{x}: %{y}')
    st.plotly_chart(fig_metrics, use_container_width=True, key="metrics_chart_final")

    # ================= FEATURE IMPORTANCE =================
    st.subheader("📈 Model Feature Importance")
    if hasattr(model, 'named_steps'):
        rf_model = list(model.named_steps.values())[-1]
    else:
        rf_model = model

    importances = rf_model.feature_importances_
    features = ["id","age","trestbps","chol","thalch","oldpeak","ca"]

    fi_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=True)
    fig_fi = px.bar(
        fi_df,
        x="Importance",
        y="Feature",
        orientation="h",
        template="plotly_dark",
        color="Importance",
        color_continuous_scale="reds",
        text="Importance"
    )
    fig_fi.update_traces(textposition='outside', hovertemplate='%{y}: %{x:.2f}')
    st.plotly_chart(fig_fi, use_container_width=True, key="feature_importance_chart_final")

    # ================= AI RECOMMENDATIONS =================
    st.divider()
    st.subheader("💡 Recommendations")
    if prob >= 70:
        st.markdown("<p style='color:red; font-weight:bold;'>⚠️ High Risk: Immediate medical consultation recommended.</p>", unsafe_allow_html=True)
    elif prob >= 40:
        st.markdown("<p style='color:orange; font-weight:bold;'>⚠️ Moderate Risk: Consider lifestyle changes and regular check-ups.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color:green; font-weight:bold;'>✅ Low Risk: Maintain a healthy lifestyle and regular monitoring.</p>", unsafe_allow_html=True)
