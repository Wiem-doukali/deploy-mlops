# app/main.py - API médicale avec modèle MLflow
from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import Optional
import json
import os
import joblib
import numpy as np

app = FastAPI(
    title="Medical Diagnosis API",
    description="API de diagnostic médical avec IA et MLflow",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# CHARGER LE MODÈLE ML
# ============================================================

model = None
model_features = [
    'fever', 'cough', 'sore_throat', 'fatigue', 'chills',
    'headache', 'nausea', 'shortness_of_breath', 'loss_of_taste', 'muscle_ache'
]

def load_model():
    global model
    try:
        model_path = "model/medical_diagnosis_model.pkl"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"✅ Modèle chargé: {model_path}")
            return True
        else:
            print(f"⚠️ Modèle non trouvé: {model_path}")
            return False
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        return False

# Charger le modèle au démarrage
@app.on_event("startup")
async def startup_event():
    load_model()

# ============================================================
# CHARGER LES DONNÉES MÉDICALES
# ============================================================

def load_medical_data():
    try:
        with open('data/symptoms_diseases.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {
            "Grippe": {
                "symptoms": ["fièvre", "toux", "fatigue", "courbatures"],
                "description": "Infection virale respiratoire"
            },
            "Rhume": {
                "symptoms": ["écoulement nasal", "éternuements", "mal de gorge"],
                "description": "Infection virale légère"
            },
            "COVID-19": {
                "symptoms": ["fièvre", "toux sèche", "perte de goût/odorat"],
                "description": "Maladie infectieuse virale"
            }
        }

medical_data = load_medical_data()

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    """Endpoint racine"""
    return {
        "message": "Medical Diagnosis API",
        "version": "2.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "endpoints": {
            "health": "/health",
            "diagnose": "/diagnose (POST)",
            "predict": "/predict (POST) - Modèle ML",
            "diseases": "/diseases",
            "symptoms": "/symptoms",
            "history": "/history",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Vérification santé de l'API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "diseases_count": len(medical_data),
        "model_loaded": model is not None,
        "environment": os.getenv("ENV", "development")
    }

@app.get("/diseases")
async def get_diseases():
    """Récupère toutes les maladies"""
    return medical_data

@app.get("/symptoms")
async def get_symptoms():
    """Récupère tous les symptômes uniques"""
    all_symptoms = set()
    for disease_info in medical_data.values():
        all_symptoms.update(disease_info.get("symptoms", []))
    
    return {
        "symptoms": sorted(list(all_symptoms)),
        "count": len(all_symptoms)
    }

@app.post("/diagnose")
async def diagnose(
    symptoms_text: str = Form(...),
    patient_age: Optional[int] = Form(None),
    patient_gender: Optional[str] = Form(None)
):
    """Diagnostic à partir de symptômes (méthode NLP)"""
    if not symptoms_text.strip():
        raise HTTPException(status_code=400, detail="Le texte des symptômes est vide")
    
    # Détection des symptômes
    symptoms_lower = symptoms_text.lower()
    detected_symptoms = []
    
    for disease_info in medical_data.values():
        for symptom in disease_info.get("symptoms", []):
            if symptom.lower() in symptoms_lower and symptom not in detected_symptoms:
                detected_symptoms.append(symptom)
    
    if not detected_symptoms:
        detected_symptoms = ["symptômes décrits dans le texte"]
    
    # Trouver les maladies correspondantes
    predictions = []
    for disease_name, disease_info in medical_data.items():
        disease_symptoms = disease_info.get("symptoms", [])
        matching_symptoms = [s for s in detected_symptoms if s in disease_symptoms]
        
        if matching_symptoms:
            confidence = min(len(matching_symptoms) / max(len(disease_symptoms), 1), 1.0)
            predictions.append({
                "disease": disease_name,
                "confidence": round(confidence, 2),
                "matching_symptoms": matching_symptoms
            })
    
    # Trier par confiance
    predictions.sort(key=lambda x: x["confidence"], reverse=True)
    
    # Déterminer la sévérité
    severity = "low"
    if any("fièvre" in s.lower() for s in detected_symptoms):
        severity = "medium"
    if "difficulté à respirer" in symptoms_lower or "essoufflement" in symptoms_lower:
        severity = "high"
    
    # Recommandations
    if severity == "high":
        recommendations = "⚠️ CONSULTEZ UN MÉDECIN URGENCEMMENT. Symptômes potentiellement graves."
    elif severity == "medium":
        recommendations = "Consultez un médecin dans les 24-48 heures. Reposez-vous et hydratez-vous."
    else:
        recommendations = "Surveillez vos symptômes. Reposez-vous et prenez du paracétamol si nécessaire."
    
    return {
        "consultation_id": int(datetime.now().timestamp()),
        "method": "NLP-based",
        "symptoms_detected": detected_symptoms,
        "patient_info": {
            "age": patient_age,
            "gender": patient_gender
        },
        "predictions": predictions[:3],
        "recommendations": recommendations,
        "severity": severity,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def predict(
    fever: int = Form(0),
    cough: int = Form(0),
    sore_throat: int = Form(0),
    fatigue: int = Form(0),
    chills: int = Form(0),
    headache: int = Form(0),
    nausea: int = Form(0),
    shortness_of_breath: int = Form(0),
    loss_of_taste: int = Form(0),
    muscle_ache: int = Form(0)
):
    """Prédiction avec le modèle MLflow entraîné"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle ML non chargé")
    
    # Préparer les features
    features = np.array([[
        fever, cough, sore_throat, fatigue, chills,
        headache, nausea, shortness_of_breath, loss_of_taste, muscle_ache
    ]])
    
    try:
        # Prédiction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = float(probabilities.max())
        
        # Mapping des classes
        disease_map = {0: "Rhume", 1: "Grippe", 2: "COVID-19", 3: "Angine"}
        predicted_disease = disease_map.get(int(prediction), "Inconnu")
        
        return {
            "prediction": int(prediction),
            "disease": predicted_disease,
            "confidence": round(confidence, 4),
            "probabilities": {
                disease_map.get(i, "Unknown"): float(p) 
                for i, p in enumerate(probabilities)
            },
            "method": "MLflow RandomForest",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur prédiction: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Infos sur le modèle chargé"""
    if model is None:
        return {"status": "Model not loaded"}
    
    return {
        "model_type": type(model).__name__,
        "model_loaded": True,
        "n_estimators": model.n_estimators if hasattr(model, 'n_estimators') else None,
        "features": model_features,
        "n_features": len(model_features),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/history")
async def get_history(limit: int = 10):
    """Historique des consultations (simulé)"""
    return {
        "total": 5,
        "limit": limit,
        "consultations": [
            {
                "id": 1,
                "timestamp": "2024-01-01T10:30:00",
                "symptoms": ["fièvre", "toux"],
                "diagnosis": "Grippe",
                "severity": "medium"
            },
            {
                "id": 2,
                "timestamp": "2024-01-02T14:20:00",
                "symptoms": ["mal de gorge", "écoulement nasal"],
                "diagnosis": "Rhume",
                "severity": "low"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)