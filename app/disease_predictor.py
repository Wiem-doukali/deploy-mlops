# app/disease_predictor.py
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
import os

class DiseasePredictor:
    def __init__(self, data_path: str = "data/symptoms_diseases.json"):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Fichier {data_path} non trouvé")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.disease_data = json.load(f)
        
        self.diseases = list(self.disease_data.keys())
        
        # Préparer les descriptions de symptômes
        self.symptom_descriptions = []
        for disease, data in self.disease_data.items():
            symptoms_list = data.get('symptoms', [])
            # Créer une description textuelle
            description = ' '.join(symptoms_list)
            self.symptom_descriptions.append(description)
        
        # Initialiser et entraîner le vectorizer TF-IDF
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,
            ngram_range=(1, 2)
        )
        
        # Transformer les descriptions
        self.symptom_vectors = self.vectorizer.fit_transform(self.symptom_descriptions)
    
    def predict_diseases(self, user_symptoms: List[str]) -> List[Tuple[str, float]]:
        """Prédit les maladies basées sur les symptômes"""
        if not user_symptoms:
            return []
        
        # Créer une description textuelle des symptômes de l'utilisateur
        user_text = ' '.join(user_symptoms)
        
        try:
            # Transformer les symptômes de l'utilisateur
            user_vector = self.vectorizer.transform([user_text])
            
            # Calculer les similarités cosinus
            similarities = cosine_similarity(user_vector, self.symptom_vectors)
            
            # Associer les maladies avec leurs scores
            disease_scores = list(zip(self.diseases, similarities[0]))
            
            # Filtrer et trier
            filtered_scores = [
                (disease, float(score)) 
                for disease, score in disease_scores 
                if score > 0.1
            ]
            
            # Trier par score décroissant
            filtered_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Retourner les 5 meilleures prédictions maximum
            return filtered_scores[:5]
            
        except Exception as e:
            print(f"Erreur lors de la prédiction: {e}")
            return []
    
    def get_disease_info(self, disease_name: str) -> dict:
        """Récupère les informations d'une maladie spécifique"""
        return self.disease_data.get(disease_name, {})
    
    def get_all_diseases(self) -> List[str]:
        """Retourne la liste de toutes les maladies"""
        return self.diseases