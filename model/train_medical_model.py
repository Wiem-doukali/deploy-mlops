# train_medical_model.py
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import os

# Configuration MLflow
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("medical-diagnosis-prediction")

def load_and_prepare_data():
    """Charge et prépare les données d'entraînement"""
    print("Chargement des données...")
    
    try:
        # Charger la base de connaissances des maladies
        with open('data/symptoms_diseases.json', 'r', encoding='utf-8') as f:
            diseases_kb = json.load(f)
        
        # Charger l'historique des consultations
        with open('data/consultation_history.json', 'r', encoding='utf-8') as f:
            consultation_history = json.load(f)
        
        print(f"Base de connaissances : {len(diseases_kb)} maladies")
        print(f"Historique : {len(consultation_history)} consultations")
        
        # Extraire tous les symptômes uniques
        all_symptoms = set()
        for disease_info in diseases_kb.values():
            all_symptoms.update(disease_info["symptoms"])
        all_symptoms = list(all_symptoms)
        
        print(f"Symptômes uniques : {len(all_symptoms)}")
        
        # Préparer les données d'entraînement
        training_data = []
        
        # 1. Utiliser l'historique des consultations
        print("\nTraitement de l'historique des consultations...")
        for consult in consultation_history:
            if consult.get("symptoms_detected") and consult.get("predictions"):
                symptoms = consult["symptoms_detected"]
                
                # Prendre la prédiction principale
                if consult["predictions"]:
                    main_pred = max(consult["predictions"], 
                                  key=lambda x: x.get("confidence", 0))
                    disease = main_pred["disease"]
                    
                    # Créer un vecteur binaire de symptômes
                    symptom_vector = [1 if symptom in symptoms else 0 
                                    for symptom in all_symptoms]
                    
                    training_data.append({
                        "symptoms_vector": symptom_vector,
                        "disease": disease,
                        "source": "history"
                    })
        
        # 2. Générer des données synthétiques
        print("Génération de données synthétiques...")
        n_synthetic_per_disease = max(50, 300 // len(diseases_kb))
        
        for disease_name, disease_info in diseases_kb.items():
            symptoms_list = disease_info["symptoms"]
            
            for i in range(n_synthetic_per_disease):
                # Sélectionner aléatoirement 60-100% des symptômes
                n_selected = np.random.randint(
                    max(1, int(len(symptoms_list) * 0.6)), 
                    len(symptoms_list) + 1
                )
                selected_symptoms = np.random.choice(
                    symptoms_list, 
                    size=n_selected, 
                    replace=False
                )
                
                # Créer le vecteur
                symptom_vector = [1 if symptom in selected_symptoms else 0 
                                for symptom in all_symptoms]
                
                training_data.append({
                    "symptoms_vector": symptom_vector,
                    "disease": disease_name,
                    "source": "synthetic"
                })
        
        # Convertir en DataFrame
        df = pd.DataFrame(training_data)
        
        print(f"\nDataset final : {len(df)} échantillons")
        print(f"   - Historique : {len(df[df['source'] == 'history'])}")
        print(f"   - Synthétique : {len(df[df['source'] == 'synthetic'])}")
        
        # Préparer X et y
        X = np.array(df['symptoms_vector'].tolist())
        y = df['disease'].values
        
        return X, y, all_symptoms, diseases_kb
        
    except FileNotFoundError as e:
        print(f"Fichier non trouvé : {e}")
        return None, None, None, None
    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}")
        return None, None, None, None

def train_model(X, y, all_symptoms, diseases_kb):
    """Entraîne et évalue le modèle"""
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nDivision des données :")
    print(f"   - Entraînement : {X_train.shape[0]} échantillons")
    print(f"   - Test : {X_test.shape[0]} échantillons")
    
    # Entraînement avec MLflow
    print("\nEntraînement du modèle Random Forest...")
    
    with mlflow.start_run(run_name=f"medical-model-{datetime.now().strftime('%Y%m%d-%H%M')}"):
        
        # Paramètres du modèle
        params = {
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'class_weight': 'balanced',
        }
        
        # Créer et entraîner le modèle
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Prédictions
        y_pred = model.predict(X_test)
        
        # Calcul des métriques
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log des paramètres et métriques dans MLflow
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'symptom': all_symptoms,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Sauvegarder le graphique d'importance des features
        plt.figure(figsize=(10, 6))
        top_10 = feature_importance.head(10)
        plt.barh(top_10['symptom'], top_10['importance'])
        plt.xlabel('Importance')
        plt.title('Top 10 Symptômes les plus importants')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('feature_importance.png')
        plt.close()
        
        # Enregistrement du modèle
        model_info = {
            'model': model,
            'symptoms_list': all_symptoms,
            'diseases_knowledge': diseases_kb,
            'class_names': model.classes_.tolist(),
            'training_date': datetime.now().isoformat(),
            'accuracy': accuracy
        }
        
        # Créer le dossier model s'il n'existe pas
        os.makedirs('model', exist_ok=True)
        
        # Sauvegarder le modèle avec joblib
        model_path = 'model/medical_model.pkl'
        joblib.dump(model_info, model_path)
        
        # Log du modèle dans MLflow
        mlflow.sklearn.log_model(model, "model")
        
        # Tags
        mlflow.set_tags({
            "environment": "development",
            "model_type": "RandomForest",
            "task": "multiclass_classification"
        })
        
        print(f"\nModèle entraîné et sauvegardé dans {model_path}")
        print(f"Accuracy : {accuracy:.4f}")
        
        # Rapport de classification
        print("\nRapport de classification :")
        print(classification_report(y_test, y_pred, target_names=model.classes_, zero_division=0))
        
        # Top 5 symptômes importants
        print("\nTop 5 symptômes les plus importants :")
        for i, row in feature_importance.head(5).iterrows():
            print(f"   {i+1}. {row['symptom']} : {row['importance']:.4f}")
        
        return model_info

def main():
    """Fonction principale"""
    print("=" * 60)
    print("ENTRAÎNEMENT DU MODÈLE DE DIAGNOSTIC MÉDICAL")
    print("=" * 60)
    
    # Étape 1: Charger et préparer les données
    X, y, all_symptoms, diseases_kb = load_and_prepare_data()
    
    if X is None or y is None:
        print("Impossible de charger les données. Vérifiez les fichiers dans data/")
        print("   Assurez-vous d'avoir :")
        print("   - data/symptoms_diseases.json")
        print("   - data/consultation_history.json")
        return
    
    # Étape 2: Entraîner le modèle
    model_info = train_model(X, y, all_symptoms, diseases_kb)
    
    if model_info:
        print("\n" + "=" * 60)
        print("MLFLOW TRACKING")
        print("=" * 60)
        print("Les métriques ont été sauvegardées dans MLflow")
        print("Pour visualiser : mlflow ui --port 5000")
        print("Ouvrez http://localhost:5000 dans votre navigateur")
        
        print("\nEntraînement terminé avec succès!")
        print("Le modèle est prêt à être utilisé dans l'API médicale.")

if __name__ == "__main__":
    main()