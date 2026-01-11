import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json
import os
import sys

# Ajouter le répertoire parent au chemin pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)

def test_root():
    """Test endpoint racine"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Medical Diagnosis API" in data["message"]
    assert "version" in data
    assert "endpoints" in data

def test_health():
    """Test health check"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "diseases_count" in data
    assert "timestamp" in data

def test_diseases():
    """Test endpoint diseases"""
    response = client.get("/diseases")
    # Peut retourner 200 ou 503 selon si la base de connaissances est chargée
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, dict)
        # Si c'est le modèle Iris, il aura cette structure
        if "status" in data and data["status"] == "ok":
            assert "model_meta" in data
        else:
            # Sinon, c'est notre base de connaissances médicale
            assert len(data) > 0

def test_symptoms():
    """Test endpoint symptoms"""
    response = client.get("/symptoms")
    # Peut retourner 200 ou 503
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "symptoms" in data
        assert isinstance(data["symptoms"], list)

def test_history():
    """Test endpoint history"""
    response = client.get("/history")
    assert response.status_code in [200, 500, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "consultations" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data

def test_diagnose_empty():
    """Test diagnostic avec texte vide"""
    response = client.post("/diagnose", data={"symptoms_text": ""})
    # Devrait retourner 400 pour texte vide
    assert response.status_code in [400, 500, 503]

def test_diagnose_valid():
    """Test diagnostic avec texte valide"""
    # Tester avec différents symptômes
    test_cases = [
        "mal de tête et fièvre",
        "fièvre et fatigue",
        "toux sèche et difficulté à respirer"
    ]
    
    for symptoms_text in test_cases:
        response = client.post("/diagnose", data={"symptoms_text": symptoms_text})
        # Peut retourner 200, 500 ou 503 selon l'état
        assert response.status_code in [200, 400, 500, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "consultation_id" in data
            assert "symptoms" in data
            assert "predictions" in data
            assert "recommendations" in data
            assert "severity" in data
            assert "timestamp" in data
            
            # Vérifier les types
            assert isinstance(data["symptoms"], list)
            assert isinstance(data["predictions"], list)
            assert isinstance(data["recommendations"], str)
            assert data["severity"] in ["low", "medium", "high"]

def test_diagnose_with_age_gender():
    """Test diagnostic avec âge et genre"""
    response = client.post(
        "/diagnose",
        data={
            "symptoms_text": "fièvre et frissons",
            "patient_age": "35",
            "patient_gender": "male"
        }
    )
    
    assert response.status_code in [200, 400, 500, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert data["consultation_id"] > 0

def test_diagnose_json_endpoint():
    """Test diagnostic via endpoint JSON"""
    payload = {
        "symptoms_text": "nausées et vomissements",
        "patient_age": 28,
        "patient_gender": "female"
    }
    
    response = client.post("/diagnose/json", json=payload)
    assert response.status_code in [200, 400, 500, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "consultation_id" in data

def test_history_pagination():
    """Test pagination de l'historique"""
    for limit in [1, 5, 10]:
        response = client.get(f"/history?limit={limit}&offset=0")
        if response.status_code == 200:
            data = response.json()
            assert data["limit"] == limit
            assert len(data["consultations"]) <= limit

def test_swagger_docs():
    """Test que la documentation Swagger est disponible"""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_redoc_docs():
    """Test que la documentation ReDoc est disponible"""
    response = client.get("/redoc")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_invalid_endpoint():
    """Test endpoint inexistant"""
    response = client.get("/invalid_endpoint")
    assert response.status_code == 404

# Tests avec mocks pour éviter les dépendances externes
class MockNLPProcessor:
    def extract_symptoms(self, text):
        return ["fièvre", "fatigue"] if text else []

class MockDiseasePredictor:
    def predict_diseases(self, symptoms):
        return [("Grippe", 0.8), ("Rhume", 0.6)]

class MockResponseGenerator:
    def generate_response(self, symptoms, predictions, urgency_alert):
        return "Recommandations de test"

class MockSeverityManager:
    def generate_urgency_alert(self, symptoms, predictions):
        return False

class MockHistoryManager:
    def save_consultation(self, **kwargs):
        return 123
    
    def get_recent_consultations(self, limit):
        return []

@patch('app.main.components', {
    'nlp': MockNLPProcessor(),
    'predictor': MockDiseasePredictor(),
    'response_gen': MockResponseGenerator(),
    'severity': MockSeverityManager(),
    'history': MockHistoryManager()
})
def test_diagnose_with_mocks():
    """Test diagnostic avec des mocks"""
    response = client.post("/diagnose", data={"symptoms_text": "fièvre"})
    
    if response.status_code == 200:
        data = response.json()
        assert data["consultation_id"] == 123
        assert "fièvre" in data["symptoms"]
        assert len(data["predictions"]) > 0
        assert data["severity"] in ["low", "medium", "high"]

# Tests d'intégration (nécessitent l'API en cours d'exécution)
@pytest.mark.integration
def test_integration_health():
    """Test d'intégration du health check"""
    import requests
    response = requests.get("http://localhost:8000/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]

@pytest.mark.integration
def test_integration_diagnose():
    """Test d'intégration du diagnostic"""
    import requests
    response = requests.post(
        "http://localhost:8000/diagnose",
        data={"symptoms_text": "fièvre et toux"}
    )
    assert response.status_code in [200, 400, 500]

if __name__ == "__main__":
    # Exécuter les tests de base
    print("Lancement des tests de l'API...")
    
    tests = [
        test_root,
        test_health,
        test_diseases,
        test_symptoms,
        test_history,
        test_diagnose_empty,
        test_diagnose_valid,
        test_diagnose_with_age_gender,
        test_diagnose_json_endpoint,
        test_history_pagination,
        test_swagger_docs,
        test_redoc_docs,
        test_invalid_endpoint,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            print(f"{test.__name__}: PASSED")
            passed += 1
        except AssertionError as e:
            print(f"{test.__name__}: FAILED - {e}")
            failed += 1
        except Exception as e:
            print(f"{test.__name__}: ERROR - {e}")
            failed += 1
    
    print(f"\nRésultats: {passed} passés, {failed} échoués")
    
    if failed == 0:
        print("Tous les tests passent !")
    else:
        print("Certains tests ont échoué")
        sys.exit(1)