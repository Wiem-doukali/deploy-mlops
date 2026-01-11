# app/drift_detector.py
# Module de détection et simulation du Data Drift

import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
import json
import os

class DriftDetector:
    """Détecte le Data Drift et Concept Drift"""
    
    def __init__(self, threshold=0.05):
        """
        Args:
            threshold: p-value threshold pour détecter le drift
        """
        self.threshold = threshold
        self.reference_data = None
        self.drift_history = []
    
    def set_reference_data(self, X_reference):
        """Définit les données de référence"""
        self.reference_data = X_reference
        print(f"✅ Reference data set: {X_reference.shape}")
    
    def detect_drift_ks_test(self, X_current, feature_names=None):
        """
        Détecte le drift avec Kolmogorov-Smirnov Test
        
        Args:
            X_current: Nouvelles données
            feature_names: Noms des features
            
        Returns:
            dict avec résultats du drift pour chaque feature
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference_data first.")
        
        n_features = X_current.shape[1]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        drift_results = {}
        
        for i, feature_name in enumerate(feature_names):
            ref_feature = self.reference_data[:, i]
            curr_feature = X_current[:, i]
            
            # Kolmogorov-Smirnov Test
            statistic, p_value = stats.ks_2samp(ref_feature, curr_feature)
            
            drift_detected = p_value < self.threshold
            
            drift_results[feature_name] = {
                "drift_detected": drift_detected,
                "p_value": float(p_value),
                "statistic": float(statistic),
                "type": "KS Test",
                "severity": self._get_severity(p_value)
            }
        
        # Calculer le drift global
        drift_count = sum(1 for r in drift_results.values() if r["drift_detected"])
        global_drift_percentage = (drift_count / len(drift_results)) * 100
        
        return {
            "features": drift_results,
            "global_drift_percentage": global_drift_percentage,
            "drift_detected": global_drift_percentage > 0,
            "timestamp": datetime.now().isoformat()
        }
    
    def detect_drift_statistical(self, X_current, feature_names=None):
        """
        Détecte le drift avec tests statistiques multiples
        
        - Mean shift detection
        - Variance shift detection
        - Distribution shift detection
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set")
        
        n_features = X_current.shape[1]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        drift_results = {}
        
        for i, feature_name in enumerate(feature_names):
            ref_feature = self.reference_data[:, i]
            curr_feature = X_current[:, i]
            
            # Test 1: Mean shift (T-test)
            t_stat, t_pvalue = stats.ttest_ind(ref_feature, curr_feature)
            mean_drift = t_pvalue < self.threshold
            
            # Test 2: Variance shift (Levene's test)
            levene_stat, levene_pvalue = stats.levene(ref_feature, curr_feature)
            variance_drift = levene_pvalue < self.threshold
            
            # Test 3: Distribution shift (KS test)
            ks_stat, ks_pvalue = stats.ks_2samp(ref_feature, curr_feature)
            dist_drift = ks_pvalue < self.threshold
            
            drift_detected = mean_drift or variance_drift or dist_drift
            
            drift_results[feature_name] = {
                "drift_detected": drift_detected,
                "mean_shift": {
                    "detected": mean_drift,
                    "p_value": float(t_pvalue),
                    "ref_mean": float(np.mean(ref_feature)),
                    "curr_mean": float(np.mean(curr_feature))
                },
                "variance_shift": {
                    "detected": variance_drift,
                    "p_value": float(levene_pvalue),
                    "ref_var": float(np.var(ref_feature)),
                    "curr_var": float(np.var(curr_feature))
                },
                "distribution_shift": {
                    "detected": dist_drift,
                    "p_value": float(ks_pvalue)
                },
                "severity": self._get_severity(min(t_pvalue, levene_pvalue, ks_pvalue))
            }
        
        return {
            "features": drift_results,
            "timestamp": datetime.now().isoformat()
        }
    
    def simulate_drift(self, X_reference, drift_type="mean", intensity=1.0):
        """
        Simule différents types de drift
        
        Args:
            X_reference: Données de référence
            drift_type: "mean", "variance", "missing_values", "outliers", "distribution"
            intensity: Intensité du drift (0.0 à 1.0)
            
        Returns:
            X_drifted: Données avec drift simulé
        """
        X_drifted = X_reference.copy().astype(float)
        
        if drift_type == "mean":
            # Shift dans la moyenne
            shift = np.std(X_drifted) * intensity * 2
            X_drifted += shift
        
        elif drift_type == "variance":
            # Augmentation de la variance
            center = np.mean(X_drifted)
            X_drifted = center + (X_drifted - center) * (1 + intensity * 2)
        
        elif drift_type == "missing_values":
            # Introduit des valeurs manquantes
            n_missing = int(len(X_drifted) * intensity * 0.2)
            indices = np.random.choice(len(X_drifted), n_missing, replace=False)
            X_drifted[indices] = np.nan
        
        elif drift_type == "outliers":
            # Ajoute des outliers
            n_outliers = int(len(X_drifted) * intensity * 0.1)
            outlier_indices = np.random.choice(len(X_drifted), n_outliers, replace=False)
            X_drifted[outlier_indices] *= (3 + intensity)
        
        elif drift_type == "distribution":
            # Change la distribution (e.g., exponentielle)
            X_drifted = np.random.exponential(np.mean(X_reference) * intensity, X_reference.shape)
        
        return X_drifted
    
    def _get_severity(self, p_value):
        """Détermine la sévérité du drift"""
        if p_value < 0.001:
            return "CRITICAL"
        elif p_value < 0.01:
            return "HIGH"
        elif p_value < 0.05:
            return "MEDIUM"
        else:
            return "LOW"
    
    def log_drift_event(self, drift_results, reason="monitoring"):
        """Enregistre un événement de drift"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "results": drift_results
        }
        self.drift_history.append(event)
        
        # Sauvegarder en JSON
        os.makedirs("logs", exist_ok=True)
        with open("logs/drift_history.json", "w") as f:
            json.dump(self.drift_history, f, indent=2)
        
        return event


# Fonction utilitaire pour charger/tester le drift
def create_drift_report(X_reference, X_current, feature_names=None):
    """
    Crée un rapport complet de drift
    
    Returns:
        dict avec analyse détaillée
    """
    detector = DriftDetector(threshold=0.05)
    detector.set_reference_data(X_reference)
    
    # Détection KS
    ks_results = detector.detect_drift_ks_test(X_current, feature_names)
    
    # Détection statistique
    stat_results = detector.detect_drift_statistical(X_current, feature_names)
    
    return {
        "ks_test": ks_results,
        "statistical_tests": stat_results,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    # Test
    np.random.seed(42)
    
    # Données de référence
    X_ref = np.random.normal(loc=0, scale=1, size=(1000, 5))
    
    # Données sans drift
    X_clean = np.random.normal(loc=0, scale=1, size=(100, 5))
    
    # Données avec drift (mean shift)
    X_drift = np.random.normal(loc=2, scale=1, size=(100, 5))
    
    # Détection
    detector = DriftDetector(threshold=0.05)
    detector.set_reference_data(X_ref)
    
    print("=" * 60)
    print("✅ SANS DRIFT")
    print("=" * 60)
    results_clean = detector.detect_drift_ks_test(X_clean)
    print(f"Global Drift: {results_clean['global_drift_percentage']:.2f}%")
    
    print("\n" + "=" * 60)
    print("⚠️ AVEC DRIFT (MEAN SHIFT)")
    print("=" * 60)
    results_drift = detector.detect_drift_ks_test(X_drift)
    print(f"Global Drift: {results_drift['global_drift_percentage']:.2f}%")
    
    print("\n" + "=" * 60)
    print("🎲 SIMULATION DE DRIFT")
    print("=" * 60)
    
    # Simule différents types de drift
    for drift_type in ["mean", "variance", "outliers"]:
        X_simulated = detector.simulate_drift(X_ref, drift_type=drift_type, intensity=0.7)
        results = detector.detect_drift_ks_test(X_simulated[:100])
        print(f"{drift_type.upper()}: {results['global_drift_percentage']:.2f}% drift detected")