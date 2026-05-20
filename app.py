import streamlit as st
import pandas as pd
import joblib

# --- إعداد الصفحة ---
st.set_page_config(page_title="ChatBox AI", layout="centered")

# --- CSS مخصص ---
st.markdown("""
<style>
body {
    background-color: #0d1117;
}

.main {
    text-align: center;
}

.title {
    font-size: 48px;
    font-weigt: bold;
    color: white;
}

.subtitle {
    color: white;
    margin-bottom: 30px;
}

.input-box {
    background-color: #161b22;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #30363d;
    margin-bottom: 15px;
}

.stButton>button {
    background: linear-gradient(90deg, #00d1b2, #00ffcc);
    color: black;
    font-size: 18px;
    font-weight: bold;
    border-radius: 30px;
    padding: 10px 40px;
    border: none;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #00ffcc, #00d1b2);
}
</style>
""", unsafe_allow_html=True)

# --- العنوان ---
st.markdown('<div class="title">🎓 Prévision de la réussite académique</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Saisissez les données de l’étudiant pour une analyse intelligente</div>', unsafe_allow_html=True)

# --- تحميل الموديل ---
@st.cache_resource
def load_model():
    models = joblib.load("student_models.pkl")
    imputer = joblib.load("imputer.pkl")
    return models, imputer

models, imputer = load_model()
model_names = list(models.keys())
default_model = "Random Forest"

# --- اختيار الموديل ---
model_name = st.selectbox("Choisir le modèle", model_names, index=model_names.index(default_model))

# --- Inputs ---
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Âge", 15, 30, 18)
    g1 = st.number_input("Semestre 1", 0.0, 20.0, 10.0)

with col2:
    absences = st.number_input("Absences", 0, 100, 0)
    study_hours = st.slider("Heures d'étude", 1.0, 8.0, 2.0, step=0.5)

# --- Bouton ---
if st.button("Analyse des données"):
    
    # ❌ g2 supprimé ici aussi
    data = pd.DataFrame([[age, absences, g1, study_hours]],
                        columns=["age", "absences", "g1", "study_hours"])

    data = pd.DataFrame(imputer.transform(data), columns=data.columns)

    model = models[model_name]
    prediction = model.predict(data)
    proba = model.predict_proba(data)[0]

    success = proba[1] * 100
    fail = proba[0] * 100

    if prediction[0] == 1:
        st.success(f"✅ Succès probable : {success:.2f}%")
    else:
        st.error(f"⚠️ Risque d'échec\nSuccès: {success:.2f}% | Échec: {fail:.2f}%")
        
