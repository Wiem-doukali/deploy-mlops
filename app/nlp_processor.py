# app/nlp_processor.py
import spacy
import re
from typing import List, Set

class NLPProcessor:
    def __init__(self):
        try:
            # Essayer de charger le modèle français
            self.nlp = spacy.load("fr_core_news_sm")
        except OSError:
            # Si le modèle n'est pas trouvé, donner des instructions
            raise ImportError(
                "Modèle spaCy français non trouvé.\n"
                "Exécutez: python -m spacy download fr_core_news_sm\n"
                "Ou installez via: pip install https://github.com/explosion/spacy-models/releases/download/fr_core_news_sm-3.7.0/fr_core_news_sm-3.7.0-py3-none-any.whl"
            )
        
        # Termes médicaux français étendus
        self.medical_terms = {
            'gorge', 'fièvre', 'toux', 'fatigue', 'nez', 'courbature',
            'tête', 'nausée', 'vomissement', 'douleur', 'mal', 'frisson',
            'éternuement', 'rhume', 'grippe', 'angine', 'migraine', 'gastro',
            'diarrhée', 'respiration', 'allergie', 'démangeaison', 'yeux',
            'abdominal', 'facial', 'sinus', 'blocage', 'ganglion', 'enrouement',
            'oreille', 'audition', 'thoracique', 'saignement', 'paralysie',
            'convulsion', 'brûlure', 'urinaire', 'articulation', 'raideur',
            'essoufflement', 'oppression', 'sifflement', 'transpiration',
            'vertige', 'étourdissement', 'palpitation', 'crampe', 'engourdissement',
            'picotement', 'faiblesse', 'perte', 'vision', 'auditif', 'olfactif',
            'goût', 'équilibre', 'coordination', 'mémoire', 'concentration',
            'sommeil', 'appétit', 'soif', 'urine', 'selles', 'constipation',
            'ballonnement', 'flatulence', 'brûlure', 'reflux', 'acidité',
            'plaie', 'coupure', 'ecchymose', 'gonflement', 'rougeur', 'chaleur',
            'démangeaison', 'desquamation', 'bouton', 'éruption', 'rougeole',
            'varicelle', 'urticaire', 'eczéma', 'psoriasis'
        }
    
    def preprocess_text(self, text: str) -> List[str]:
        """Prétraite le texte et retourne les tokens"""
        if not text or not isinstance(text, str):
            return []
            
        # Nettoyage du texte
        text = re.sub(r'[^\w\sàâäéèêëîïôöùûüç]', ' ', text.lower())
        doc = self.nlp(text)
        
        # Extraction des tokens
        tokens = []
        for token in doc:
            if (not token.is_stop and 
                not token.is_punct and 
                token.is_alpha and 
                len(token.text) > 1):
                # Utiliser le lemme (forme de base)
                lemma = token.lemma_.lower()
                tokens.append(lemma)
        
        return tokens
    
    def extract_symptoms(self, text: str) -> List[str]:
        """Extrait les symptômes du texte"""
        if not text:
            return []
            
        tokens = self.preprocess_text(text)
        detected_symptoms = []
        text_lower = text.lower()
        
        # Patterns médicaux complets
        medical_patterns = {
            'mal de gorge': ['gorge', 'mal'],
            'nez qui coule': ['nez', 'couler'],
            'nez bouché': ['nez', 'bouché'],
            'maux de tête': ['tête', 'mal', 'migraine'],
            'courbatures': ['courbature'],
            'nausées': ['nausée'],
            'vomissements': ['vomissement'],
            'fièvre': ['fièvre'],
            'toux': ['toux'],
            'fatigue': ['fatigue'],
            'difficulté à respirer': ['respiration', 'difficulté', 'essoufflement'],
            'essoufflement': ['essoufflement'],
            'oppression thoracique': ['oppression', 'thoracique'],
            'douleur thoracique': ['thoracique', 'douleur'],
            'yeux qui piquent': ['yeux', 'piquer'],
            'démangeaisons': ['démangeaison'],
            'douleur abdominale': ['abdominal', 'douleur'],
            'douleur faciale': ['facial', 'douleur'],
            'frissons': ['frisson'],
            'diarrhée': ['diarrhée'],
            'ganglions': ['ganglion'],
            'enrouement': ['enrouement'],
            'perte de voix': ['voix', 'perte'],
            'douleur oreille': ['oreille', 'douleur'],
            'baisse audition': ['audition', 'baisse'],
            'saignement abondant': ['saignement', 'abondant'],
            'paralysie': ['paralysie'],
            'crise convulsive': ['convulsive', 'crise'],
            'brûlure grave': ['brûlure', 'grave'],
            'brûlures urinaires': ['urinaire', 'brûlure'],
            'douleur articulation': ['articulation', 'douleur'],
            'raideur': ['raideur'],
            'vertiges': ['vertige'],
            'étourdissements': ['étourdissement'],
            'palpitations': ['palpitation'],
            'transpiration excessive': ['transpiration', 'excessive'],
            'perte de conscience': ['conscience', 'perte'],
            'trouble de la vision': ['vision', 'trouble'],
            'trouble de l\'équilibre': ['équilibre', 'trouble']
        }
        
        # 1. Recherche directe dans le texte original
        for pattern in medical_patterns.keys():
            if pattern in text_lower:
                detected_symptoms.append(pattern)
        
        # 2. Recherche par combinaison de tokens
        for pattern, keywords in medical_patterns.items():
            if pattern not in detected_symptoms:
                # Vérifier si tous les mots-clés sont présents
                if all(any(keyword in token for token in tokens) for keyword in keywords):
                    detected_symptoms.append(pattern)
        
        # 3. Détection des termes médicaux isolés
        for token in tokens:
            if token in self.medical_terms:
                symptom_name = self.get_symptom_name_from_token(token)
                if symptom_name and symptom_name not in detected_symptoms:
                    detected_symptoms.append(symptom_name)
        
        # 4. Détection par expressions régulières
        regex_patterns = {
            r'température.*?(\d+[.,]?\d*)': 'fièvre',
            r'fièvre.*?(\d+[.,]?\d*)': 'fièvre avec température',
            r'depuis.*?(\d+).*?(heure|jour|semaine)': 'durée spécifiée',
            r'douleur.*?(forte|sévère|intense)': 'douleur sévère',
            r'douleur.*?(légère|modérée)': 'douleur modérée'
        }
        
        for pattern, symptom in regex_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                if symptom not in detected_symptoms:
                    detected_symptoms.append(symptom)
        
        return list(set(detected_symptoms))
    
    def get_symptom_name_from_token(self, token: str) -> str:
        """Convertit un token en nom de symptôme lisible"""
        symptom_map = {
            'fièvre': 'fièvre',
            'toux': 'toux',
            'fatigue': 'fatigue',
            'nausée': 'nausées',
            'vomissement': 'vomissements',
            'frisson': 'frissons',
            'diarrhée': 'diarrhée',
            'démangeaison': 'démangeaisons',
            'courbature': 'courbatures',
            'vertige': 'vertiges',
            'palpitation': 'palpitations'
        }
        return symptom_map.get(token, token)