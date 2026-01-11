# app/utils.py
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_json_file(filepath: str) -> Dict:
    """Charge un fichier JSON"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Erreur chargement {filepath}: {e}")
        return {}

def save_json_file(data: Any, filepath: str) -> bool:
    """Sauvegarde des données en JSON"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"Erreur sauvegarde {filepath}: {e}")
        return False

def validate_symptoms(symptoms: list) -> list:
    """Nettoie et valide les symptômes"""
    if not symptoms:
        return []
    
    # Nettoyage basique
    cleaned = []
    for symptom in symptoms:
        if isinstance(symptom, str):
            symptom = symptom.strip().lower()
            if symptom and symptom not in cleaned:
                cleaned.append(symptom)
    
    return cleaned

def calculate_confidence(symptoms_matched: int, total_symptoms: int) -> float:
    """Calcule un score de confiance"""
    if total_symptoms == 0:
        return 0.0
    confidence = symptoms_matched / total_symptoms
    return min(confidence, 1.0)

def format_timestamp(timestamp: str = None) -> str:
    """Formate un timestamp"""
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    return timestamp

def get_environment() -> str:
    """Retourne l'environnement d'exécution"""
    return os.getenv('ENVIRONMENT', 'development')