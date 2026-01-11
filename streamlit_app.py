# streamlit_app.py
# Application Streamlit pour tester et visualiser l'API Medical Diagnosis

import streamlit as st
import requests
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Medical Diagnosis MLOps",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL de l'API (locale ou Render)
API_URL = st.sidebar.selectbox(
    "Sélectionne l'API",
    [
        "http://localhost:8000",
        "https://medical-api-v1.onrender.com"
    ]
)

st.sidebar.markdown("---")

# ============================================================
# TITRE ET INTRO
# ============================================================

col1, col2 = st.columns([3, 1])

with col1:
    st.title("🏥 Medical Diagnosis MLOps Dashboard")
    st.markdown("**Diagnostic médical intelligent avec ML et Monitoring**")

with col2:
    # Status API
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            st.success("✅ API Online")
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ API Offline")

st.markdown("---")

# ============================================================
# NAVIGATION
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🩺 Diagnostic",
    "🤖 Prédiction ML",
    "📊 Maladies & Symptômes",
    "📈 Drift Detection",
    "📝 Historique",
    "ℹ️ Info API"
])

# ============================================================
# TAB 1: DIAGNOSTIC (NLP)
# ============================================================

with tab1:
    st.header("🩺 Diagnostic par Symptômes")
    st.markdown("Entrez vos symptômes et obtenez un diagnostic via NLP")
    
    col1, col2 = st.columns(2)
    
    with col1:
        symptoms = st.text_area(
            "Décrivez vos symptômes",
            placeholder="Ex: fièvre, toux, fatigue, mal de gorge",
            height=100
        )
        age = st.number_input("Âge", min_value=0, max_value=120, value=30)
    
    with col2:
        gender = st.selectbox("Genre", ["Male", "Female", "Other"])
        st.empty()
    
    if st.button("🔍 Obtenir un diagnostic", use_container_width=True):
        if symptoms.strip():
            try:
                response = requests.post(
                    f"{API_URL}/diagnose",
                    data={
                        "symptoms_text": symptoms,
                        "patient_age": age,
                        "patient_gender": gender
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success("✅ Diagnostic obtenu!")
                    
                    # Résultats
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Sévérité", result["severity"].upper())
                    with col2:
                        st.metric("Symptômes détectés", len(result["symptoms_detected"]))
                    with col3:
                        st.metric("Maladies proposées", len(result["predictions"]))
                    
                    # Recommandations
                    st.info(f"📋 {result['recommendations']}")
                    
                    # Prédictions
                    st.subheader("🔬 Maladies possibles")
                    
                    for pred in result["predictions"]:
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.write(f"**{pred['disease']}**")
                        with col2:
                            confidence = int(pred['confidence'] * 100)
                            st.write(f"{confidence}% de confiance")
                        with col3:
                            st.progress(pred['confidence'])
                    
                    # Détails
                    with st.expander("📊 Détails"):
                        st.json(result)
                else:
                    st.error("Erreur API")
            
            except requests.exceptions.ConnectionError:
                st.error("❌ Impossible de se connecter à l'API")
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")
        else:
            st.warning("⚠️ Veuillez décrire vos symptômes")

# ============================================================
# TAB 2: PRÉDICTION ML
# ============================================================

with tab2:
    st.header("🤖 Prédiction ML avec Modèle Entraîné")
    st.markdown("Utilisez le modèle MLflow entraîné pour prédire")
    
    st.info("Sélectionnez la présence/absence de chaque symptôme (1 = présent, 0 = absent)")
    
    # Features
    features = {
        'fever': st.checkbox('Fièvre', value=0),
        'cough': st.checkbox('Toux', value=0),
        'sore_throat': st.checkbox('Mal de gorge', value=0),
        'fatigue': st.checkbox('Fatigue', value=0),
        'chills': st.checkbox('Frissons', value=0),
        'headache': st.checkbox('Maux de tête', value=0),
        'nausea': st.checkbox('Nausées', value=0),
        'shortness_of_breath': st.checkbox('Essoufflement', value=0),
        'loss_of_taste': st.checkbox('Perte de goût', value=0),
        'muscle_ache': st.checkbox('Courbatures', value=0)
    }
    
    if st.button("🎯 Prédire avec le modèle ML", use_container_width=True):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                data=features,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                
                st.success("✅ Prédiction obtenue!")
                
                # Résultats
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Maladie prédite", result["disease"])
                with col2:
                    confidence = int(result["confidence"] * 100)
                    st.metric("Confiance", f"{confidence}%")
                with col3:
                    st.metric("Méthode", "RandomForest")
                
                # Probabilities
                st.subheader("📊 Probabilités par classe")
                
                probs_df = pd.DataFrame(
                    list(result["probabilities"].items()),
                    columns=["Maladie", "Probabilité"]
                )
                
                fig = px.bar(
                    probs_df,
                    x="Maladie",
                    y="Probabilité",
                    color="Probabilité",
                    color_continuous_scale="RdYlGn"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Détails
                with st.expander("📋 Détails complets"):
                    st.json(result)
            else:
                st.error("Erreur API")
        
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")

# ============================================================
# TAB 3: MALADIES ET SYMPTÔMES
# ============================================================

with tab3:
    st.header("📊 Base de Maladies et Symptômes")
    
    col1, col2 = st.columns(2)
    
    # Maladies
    with col1:
        st.subheader("🦠 Maladies disponibles")
        
        try:
            response = requests.get(f"{API_URL}/diseases", timeout=5)
            if response.status_code == 200:
                diseases = response.json()
                st.metric("Total", len(diseases))
                
                for disease_name in list(diseases.keys())[:10]:
                    with st.expander(f"🏥 {disease_name}"):
                        disease_info = diseases[disease_name]
                        st.write(f"**Description**: {disease_info.get('description', 'N/A')}")
                        st.write(f"**Sévérité**: {disease_info.get('severity', 'N/A')}")
                        st.write(f"**Symptômes**: {', '.join(disease_info.get('symptoms', []))}")
        except:
            st.error("Impossible de charger les maladies")
    
    # Symptômes
    with col2:
        st.subheader("🔍 Symptômes uniques")
        
        try:
            response = requests.get(f"{API_URL}/symptoms", timeout=5)
            if response.status_code == 200:
                symptoms_data = response.json()
                st.metric("Total", symptoms_data["count"])
                
                symptoms_list = symptoms_data["symptoms"]
                
                # Affiche par groupes
                col_size = 3
                for i in range(0, len(symptoms_list), col_size):
                    cols = st.columns(col_size)
                    for j, col in enumerate(cols):
                        if i + j < len(symptoms_list):
                            with col:
                                st.write(f"✓ {symptoms_list[i + j]}")
        except:
            st.error("Impossible de charger les symptômes")

# ============================================================
# TAB 4: DRIFT DETECTION
# ============================================================

with tab4:
    st.header("📈 Data Drift Detection & Monitoring")
    st.markdown("Détectez et simulez le Data Drift dans vos données")
    
    tab4_1, tab4_2 = st.tabs(["🔍 Détection", "🎲 Simulation"])
    
    with tab4_1:
        st.subheader("Détectez le Drift entre deux datasets")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Données de référence**")
            ref_data = st.text_area(
                "Entrez les données de référence (JSON)",
                value="[[1,2,3],[4,5,6],[7,8,9]]",
                height=100,
                key="ref_data"
            )
        
        with col2:
            st.write("**Données actuelles**")
            curr_data = st.text_area(
                "Entrez les données actuelles (JSON)",
                value="[[1,2,3],[4,5,6],[7,8,9]]",
                height=100,
                key="curr_data"
            )
        
        if st.button("🔍 Détecter le Drift"):
            try:
                response = requests.post(
                    f"{API_URL}/drift/detect",
                    data={
                        "reference_data": ref_data,
                        "current_data": curr_data
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success("✅ Analyse complète!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Drift détecté", "OUI" if result["drift_detected"] else "NON")
                    with col2:
                        st.metric("% Drift", f"{result['global_drift_percentage']:.2f}%")
                    with col3:
                        st.metric("Sévérité", result["severity"])
                    
                    with st.expander("📊 Détails par feature"):
                        for feature, details in result["features"].items():
                            st.write(f"**{feature}**")
                            st.json(details)
                else:
                    st.error("Erreur API")
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")
    
    with tab4_2:
        st.subheader("Simulez différents types de Drift")
        
        data_input = st.text_area(
            "Données",
            value="[[1,2,3],[4,5,6],[7,8,9]]",
            height=80
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            drift_type = st.selectbox(
                "Type de drift",
                ["mean", "variance", "outliers", "distribution", "missing_values"]
            )
        
        with col2:
            intensity = st.slider("Intensité", 0.0, 1.0, 0.5)
        
        if st.button("🎲 Simuler le Drift"):
            try:
                response = requests.post(
                    f"{API_URL}/drift/simulate",
                    data={
                        "data": data_input,
                        "drift_type": drift_type,
                        "intensity": intensity
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success("✅ Simulation réalisée!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Moyenne originale", f"{result['original_mean']:.2f}")
                        st.metric("Std originale", f"{result['original_std']:.2f}")
                    
                    with col2:
                        st.metric("Moyenne après drift", f"{result['drifted_mean']:.2f}")
                        st.metric("Std après drift", f"{result['drifted_std']:.2f}")
                    
                    st.info(result["message"])
                else:
                    st.error("Erreur API")
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")

# ============================================================
# TAB 5: HISTORIQUE
# ============================================================

with tab5:
    st.header("📝 Historique")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Consultations")
        try:
            response = requests.get(f"{API_URL}/history", timeout=5)
            if response.status_code == 200:
                history = response.json()
                st.metric("Total consultations", history["total"])
                
                for consultation in history["consultations"][:5]:
                    with st.expander(f"ID: {consultation['id']}"):
                        st.write(f"**Date**: {consultation['timestamp']}")
                        st.write(f"**Diagnostic**: {consultation['diagnosis']}")
                        st.write(f"**Sévérité**: {consultation['severity']}")
                        st.write(f"**Symptômes**: {', '.join(consultation['symptoms'])}")
        except:
            st.warning("Impossible de charger l'historique")
    
    with col2:
        st.subheader("📊 Drift History")
        try:
            response = requests.get(f"{API_URL}/drift/history", timeout=5)
            if response.status_code == 200:
                drift_history = response.json()
                st.metric("Événements drift", drift_history["total_events"])
                st.info(f"Derniers {min(3, drift_history['total_events'])} événements détectés")
        except:
            st.info("Pas d'historique drift encore")

# ============================================================
# TAB 6: INFO API
# ============================================================

with tab6:
    st.header("ℹ️ Informations API")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📡 Endpoints disponibles")
        
        endpoints = {
            "GET": ["/health", "/diseases", "/symptoms", "/history", "/docs"],
            "POST": ["/diagnose", "/predict", "/drift/detect", "/drift/simulate"]
        }
        
        for method, eps in endpoints.items():
            st.write(f"**{method}**")
            for ep in eps:
                st.code(ep, language="text")
    
    with col2:
        st.subheader("🔧 Configuration")
        
        try:
            response = requests.get(f"{API_URL}/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                st.json(health)
        except:
            st.error("API unavailable")
    
    st.markdown("---")
    st.markdown("""
    ### 🏗️ Architecture
    
    - **Backend**: FastAPI
    - **ML**: scikit-learn, MLflow
    - **Drift Detection**: scipy, numpy
    - **Deployment**: Docker + Render
    - **CI/CD**: GitHub Actions
    
    ### 📚 Documentation
    - Swagger UI: `/docs`
    - ReDoc: `/redoc`
    """)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>🏥 <strong>Medical Diagnosis MLOps Dashboard</strong></p>
    <p>Powered by FastAPI, MLflow, Docker & Streamlit</p>
    <p>✅ Module 2, 3, 4, 5, 6 Complétés!</p>
</div>
""", unsafe_allow_html=True)