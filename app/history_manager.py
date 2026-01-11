# app/history_manager.py
import json
import os
from datetime import datetime
from typing import List, Dict, Any

class HistoryManager:
    def __init__(self, history_file: str = "data/consultation_history.json"):
        self.history_file = history_file
        self._ensure_history_file()
    
    def _ensure_history_file(self):
        """Crée le fichier d'historique s'il n'existe pas"""
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        if not os.path.exists(self.history_file):
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            print(f"Fichier {self.history_file} créé")
    
    def save_consultation(self, user_input: str, symptoms: List[str], 
                         predictions: List[tuple], response: str):
        """Sauvegarde une consultation dans l'historique"""
        try:
            entry = {
                "id": len(self.get_history()) + 1,
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input[:500],  # Limiter la taille
                "symptoms_detected": symptoms,
                "predictions": [
                    {"disease": disease, "confidence": float(score)} 
                    for disease, score in predictions[:3]  # Garder seulement les 3 meilleures
                ],
                "response_preview": response[:200] + "..." if len(response) > 200 else response
            }
            
            history = self.get_history()
            history.append(entry)
            
            # Garder seulement les 100 dernières consultations
            if len(history) > 100:
                history = history[-100:]
            
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            
            print(f"Consultation sauvegardée (ID: {entry['id']})")
            
        except Exception as e:
            print(f"Erreur sauvegarde consultation: {e}")
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Récupère l'historique complet"""
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Erreur lecture historique: {e}")
            return []
    
    def get_recent_consultations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Récupère les consultations récentes"""
        history = self.get_history()
        return history[-limit:][::-1]  # Inverser pour avoir les plus récents en premier
    
    def clear_history(self):
        """Efface tout l'historique"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            print("Historique effacé")
            return True
        except Exception as e:
            print(f"Erreur effacement historique: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne des statistiques sur l'historique"""
        history = self.get_history()
        
        if not history:
            return {
                "total_consultations": 0,
                "last_consultation": None,
                "common_symptoms": [],
                "common_diseases": []
            }
        
        # Compter les symptômes fréquents
        symptom_counter = {}
        disease_counter = {}
        
        for consult in history:
            for symptom in consult.get("symptoms_detected", []):
                symptom_counter[symptom] = symptom_counter.get(symptom, 0) + 1
            
            for prediction in consult.get("predictions", []):
                disease = prediction.get("disease", "")
                if disease:
                    disease_counter[disease] = disease_counter.get(disease, 0) + 1
        
        # Trier par fréquence
        common_symptoms = sorted(symptom_counter.items(), key=lambda x: x[1], reverse=True)[:5]
        common_diseases = sorted(disease_counter.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_consultations": len(history),
            "last_consultation": history[-1]["timestamp"] if history else None,
            "common_symptoms": common_symptoms,
            "common_diseases": common_diseases
        }