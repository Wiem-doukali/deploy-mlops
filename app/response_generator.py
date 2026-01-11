# app/response_generator.py
import json
from typing import List, Tuple, Dict, Any
import os

class ResponseGenerator:
    def __init__(self, data_path: str = "data/symptoms_diseases.json"):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Fichier {data_path} non trouvé")
            
        with open(data_path, 'r', encoding='utf-8') as f:
            self.disease_data: Dict[str, Any] = json.load(f)
    
    def generate_response(self, symptoms: List[str], predictions: List[Tuple[str, float]], 
                         urgency_alert: str = "") -> str:
        """Génère une réponse naturelle et empathique"""
        response_parts = []
        
        # Ajouter l'alerte d'urgence en premier si présente
        if urgency_alert:
            response_parts.append(urgency_alert)
            response_parts.append("\n" + "="*50 + "\n")
        
        # Introduction personnalisée
        if not symptoms and not predictions:
            response_parts.append("**Je note votre description.** ")
            response_parts.append("\nJe n'ai pas pu identifier clairement des symptômes spécifiques. ")
            response_parts.append("\n**Pour une analyse plus précise :**")
            response_parts.append("• **Soyez plus spécifique** sur ce que vous ressentez")
            response_parts.append("• **Mentionnez la durée** des symptômes") 
            response_parts.append("• **Précisez la localisation** des douleurs")
            response_parts.append("• **Décrivez l'intensité** (léger, modéré, sévère)")
            response_parts.append("\n*Exemple optimal :* « J'ai une fièvre à 38.5°C depuis 2 jours avec des maux de tête pulsatile »")
            return "\n".join(response_parts)
        
        # Si symptômes mais pas de prédictions
        if symptoms and not predictions:
            response_parts.append("**Merci pour votre description.**")
            response_parts.append(f"\nJ'ai identifié {len(symptoms)} symptôme(s) :")
            for i, symptom in enumerate(symptoms, 1):
                response_parts.append(f"{i}. {symptom}")
            
            response_parts.append("\n**Cependant, je n'ai pas pu faire de correspondance claire avec une maladie spécifique.**")
            response_parts.append("\n**Recommandations :**")
            response_parts.append("• Consultez un médecin pour un examen complet")
            response_parts.append("• Surveillez l'évolution de vos symptômes")
            response_parts.append("• Notez tout nouveau symptôme qui apparaîtrait")
            return "\n".join(response_parts)
        
        # Réponse avec prédictions
        response_parts.append("**Merci pour votre description détaillée.**")
        response_parts.append(f"\nJ'ai identifié **{len(symptoms)} symptôme(s)** et voici mon analyse préliminaire :")
        
        # Afficher les symptômes détectés
        if symptoms:
            response_parts.append("\n**Symptômes détectés :**")
            for i, symptom in enumerate(symptoms, 1):
                response_parts.append(f"• {symptom}")
        
        # Maladies détectées avec format amélioré
        response_parts.append(f"\n**Analyses possibles (par ordre de probabilité) :**")
        
        for i, (disease, score) in enumerate(predictions, 1):
            disease_info = self.disease_data.get(disease, {})
            advice = disease_info.get('advice', 'Consultez un médecin pour un diagnostic précis.')
            severity = disease_info.get('severity', 'inconnue')
            common_symptoms = disease_info.get('symptoms', [])
            
            confidence_percentage = min(score * 100, 99)
            
            # Icônes et couleurs selon la sévérité
            severity_config = {
                "légère": {"icon": "🟢", "color": "green", "emoji": "✅"},
                "modérée": {"icon": "🟡", "color": "orange", "emoji": "⚠️"},
                "urgente": {"icon": "🟠", "color": "darkorange", "emoji": "🚨"},
                "critique": {"icon": "🔴", "color": "red", "emoji": "🆘"},
                "inconnue": {"icon": "⚪", "color": "gray", "emoji": "❓"}
            }
            
            config = severity_config.get(severity, severity_config["inconnue"])
            
            response_parts.append(f"\n---")
            response_parts.append(f"**{config['emoji']} {i}. {disease.upper()}**")
            response_parts.append(f"**Niveau de confiance :** {confidence_percentage:.0f}%")
            response_parts.append(f"**Niveau de gravité :** {severity.title()} {config['icon']}")
            
            if common_symptoms:
                response_parts.append(f"**Symptômes typiques :**")
                for symptom in common_symptoms[:5]:  # Limiter à 5 symptômes
                    response_parts.append(f"  • {symptom}")
                if len(common_symptoms) > 5:
                    response_parts.append(f"  • ... et {len(common_symptoms) - 5} autre(s)")
            
            response_parts.append(f"**Conseils pratiques :** {advice}")
        
        # Recommandations personnalisées selon les symptômes
        response_parts.append(f"\n---")
        response_parts.append("**RECOMMANDATIONS PERSONNALISÉES**")
        
        # Conseils spécifiques par type de symptôme
        if any(s in ' '.join(symptoms).lower() for s in ["fièvre", "frissons", "température"]):
            response_parts.append("\n**Pour la fièvre :**")
            response_parts.append("• **Surveillance** : Prenez votre température 3 fois par jour")
            response_parts.append("• **Hydratation** : Buvez au moins 2L d'eau par jour")
            response_parts.append("• **Repos** : Évitez les efforts physiques importants")
            response_parts.append("• **Comfort** : Portez des vêtements légers, aérez la pièce")
        
        if any(s in ' '.join(symptoms).lower() for s in ["toux", "difficulté à respirer", "essoufflement", "oppression"]):
            response_parts.append("\n**Pour les symptômes respiratoires :**")
            response_parts.append("• **Environnement** : Évitez tabac, pollution, air froid")
            response_parts.append("• **Expectorations** : Buvez des boissons chaudes (tisanes, bouillon)")
            response_parts.append("• **Respiration** : Surélevez votre tête la nuit avec des oreillers")
            response_parts.append("• **Humidité** : Utilisez un humidificateur si l'air est sec")
        
        if any(s in ' '.join(symptoms).lower() for s in ["nausées", "vomissements", "diarrhée", "gastro"]):
            response_parts.append("\n**Pour les troubles digestifs :**")
            response_parts.append("• **Alimentation** : Diète hydrique (riz blanc, carottes cuites, bouillon)")
            response_parts.append("• **Surveillance** : Signes de déshydratation (bouche sèche, urines foncées)")
            response_parts.append("• **Évitez** : Laitages, fibres, aliments gras, café, alcool")
            response_parts.append("• **Fréquence** : Mangez de petites quantités fréquemment")
        
        if any(s in ' '.join(symptoms).lower() for s in ["douleur", "mal", "courbature"]):
            response_parts.append("\n**Pour la gestion de la douleur :**")
            response_parts.append("• **Repos** : Évitez les activités qui aggravent la douleur")
            response_parts.append("• **Chaud/Froid** : Appliquez de la glace pour les inflammations, du chaud pour les raideurs")
            response_parts.append("• **Position** : Adoptez une position confortable")
            response_parts.append("• **Médicaments** : Ne prenez que ceux prescrits ou conseillés par un pharmacien")
        
        # Conseils généraux pour tous
        response_parts.append(f"\n---")
        response_parts.append("**CONSEILS GÉNÉRAUX**")
        response_parts.append("• **Suivi** : Notez l'évolution de vos symptômes dans un carnet")
        response_parts.append("• **Médicaments** : Ne prenez pas d'auto-médication sans avis professionnel")
        response_parts.append("• **Communication** : Informez votre entourage de votre état")
        response_parts.append("• **Temps** : Accordez à votre corps le temps de récupérer")
        response_parts.append("• **Nutrition** : Maintenez une alimentation équilibrée autant que possible")
        
        # Avertissement médical renforcé
        response_parts.append(f"\n---")
        response_parts.append("**AVERTISSEMENT MÉDICAL IMPORTANT**")
        response_parts.append("• **Ceci est une aide préliminaire,** pas un diagnostic médical")
        response_parts.append("• **Consultez un professionnel de santé** pour un diagnostic précis")
        response_parts.append("• **En cas d'aggravation,** contactez immédiatement le 15 (SAMU)")
        response_parts.append("• **Préparez votre consultation** avec la liste de vos symptômes")
        response_parts.append("• **Mentionnez** tous les médicaments que vous prenez")
        response_parts.append("• **Historique** : Notez vos antécédents médicaux importants")
        
        # Signature
        response_parts.append(f"\n---")
        response_parts.append("*Je vous souhaite un bon rétablissement.*")
        response_parts.append("*L'équipe MedBot* ")
        
        return "\n".join(response_parts)