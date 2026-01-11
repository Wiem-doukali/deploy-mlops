# app/severity_manager.py
import json
from typing import List, Tuple, Dict

class SeverityManager:
    def __init__(self, data_path: str = "data/symptoms_diseases.json"):
        self.urgent_symptoms = {
            "difficulté à respirer": "URGENT",
            "douleur thoracique": "URGENT",
            "saignement abondant": "URGENT", 
            "paralysie": "URGENT",
            "perte de conscience": "URGENT",
            "brûlure grave": "URGENT",
            "crise convulsive": "URGENT",
            "douleur abdominale sévère": "URGENT",
            "vomissement sang": "URGENT",
            "gonflement visage": "URGENT",
            "oppression": "URGENT",
            "essoufflement": "URGENT",
            "sifflement respiratoire": "URGENT",
            "faiblesse soudaine": "URGENT",
            "trouble de la parole": "URGENT",
            "vision trouble soudaine": "URGENT",
            "maux de tête violents": "URGENT"
        }
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.disease_data = json.load(f)
        except FileNotFoundError:
            print(f"Fichier {data_path} non trouvé")
            self.disease_data = {}
        except json.JSONDecodeError:
            print(f"Erreur de décodage JSON dans {data_path}")
            self.disease_data = {}
    
    def check_symptom_severity(self, symptoms: List[str]) -> List[Tuple[str, str]]:
        """Vérifie la gravité des symptômes détectés"""
        urgent_found = []
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            for urgent_symptom, level in self.urgent_symptoms.items():
                if urgent_symptom in symptom_lower:
                    urgent_found.append((symptom, level))
                    break
        
        return urgent_found
    
    def get_disease_severity(self, disease: str) -> str:
        """Retourne la sévérité d'une maladie"""
        disease_lower = disease.lower()
        for disease_name, data in self.disease_data.items():
            if disease_lower in disease_name.lower():
                return data.get("severity", "inconnue")
        return "inconnue"
    
    def generate_urgency_alert(self, symptoms: List[str], predictions: List[Tuple[str, float]]) -> str:
        """Génère une alerte si situation urgente"""
        urgent_symptoms = self.check_symptom_severity(symptoms)
        
        if urgent_symptoms:
            alert_parts = ["**ALERTE URGENCE MÉDICALE**\n"]
            alert_parts.append("**Symptômes urgents détectés :**")
            for symptom, level in urgent_symptoms:
                alert_parts.append(f"• {symptom} ({level})")
            
            alert_parts.append("\n**ACTION REQUISE :**")
            alert_parts.append("1. **COMPOSEZ LE 15 IMMÉDIATEMENT**")
            alert_parts.append("2. **Ne conduisez pas** vous-même à l'hôpital")
            alert_parts.append("3. **Restez calme** et suivez les instructions")
            alert_parts.append("4. **Prévenez quelqu'un** de votre situation")
            alert_parts.append("5. **Préparez vos papiers** d'identité et carte vitale")
            
            return "\n".join(alert_parts)
        
        # Vérifier aussi la sévérité des maladies prédites
        for disease, score in predictions:
            if score > 0.3:  # Seuil de confiance
                severity = self.get_disease_severity(disease)
                if severity in ["urgente", "critique"]:
                    return (f"**Alerte** : La condition '{disease}' peut être grave. "
                           f"Consultez un médecin rapidement ou contactez le 15 pour avis.")
        
        return ""