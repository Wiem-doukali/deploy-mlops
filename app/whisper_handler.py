# app/whisper_handler.py - VERSION CORRECTE AVEC ARABE
import whisper
import tempfile
import os
from typing import Optional, Tuple
import numpy as np
import warnings
import time
from pydub import AudioSegment
import io

class WhisperTranscriber:
    def __init__(self, model_size: str = "base"):
        """
        Initialise le modèle Whisper
        Options: tiny, base, small, medium, large
        """
        print(f"🔍 Chargement du modèle Whisper {model_size}...")
        try:
            # Supprimer les avertissements
            warnings.filterwarnings("ignore")
            
            # Charger le modèle
            self.model = whisper.load_model(model_size)
            print(f"Modèle Whisper '{model_size}' chargé avec succès!")
            
            # Options disponibles
            self.available_models = ["tiny", "base", "small", "medium", "large"]
            
        except Exception as e:
            print(f"Erreur chargement Whisper: {e}")
            # Fallback sur un modèle plus petit
            try:
                self.model = whisper.load_model("tiny")
                print("Modèle 'tiny' chargé en fallback")
            except:
                self.model = None
                print("Impossible de charger aucun modèle Whisper")
    
    def transcribe_audio_file(self, audio_path: str, language: str = 'fr') -> str:
        """
        Transcrit un fichier audio en texte
        Paramètres:
            audio_path: chemin vers le fichier audio
            language: langue pour la transcription ('fr', 'ar', 'en', etc.)
        """
        if self.model is None:
            return "Modèle Whisper non disponible. Veuillez vérifier l'installation."
        
        try:
            if not os.path.exists(audio_path):
                return f"Fichier non trouvé: {audio_path}"
            
            # Vérifier la taille du fichier
            file_size = os.path.getsize(audio_path) / (1024 * 1024)  # en MB
            if file_size > 50:  # Limite à 50MB
                return f"Fichier trop volumineux ({file_size:.1f}MB). Maximum: 50MB."
            
            print(f"Transcription de {audio_path} en {language}...")
            start_time = time.time()
            
            # Prompt initial selon la langue
            if language == 'fr':
                initial_prompt = "Transcription médicale française. Symptômes, douleurs, fièvre, toux, fatigue."
            elif language == 'ar':
                initial_prompt = "نص طبي باللغة العربية. أعراض، آلام، حمى، سعال، تعب."
            elif language == 'en':
                initial_prompt = "Medical transcription in English. Symptoms, pain, fever, cough, fatigue."
            else:
                initial_prompt = "Medical transcription."
            
            # Options de transcription
            result = self.model.transcribe(
                audio_path,
                language=language,           # Langue (peut être 'fr', 'ar', 'en', etc.)
                task='transcribe',           # Transcription (pas traduction)
                fp16=False,                  # Important pour CPU
                temperature=0.0,             # Pour plus de cohérence
                best_of=5,                   # Meilleurs résultats
                beam_size=5,                 # Taille du beam search
                patience=1.0,                # Patience pour le décodage
                length_penalty=1.0,          # Pénalité de longueur
                suppress_tokens="-1",        # Ne supprime pas les tokens communs
                initial_prompt=initial_prompt,  # Prompt initial selon la langue
                condition_on_previous_text=True,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6
            )
            
            elapsed_time = time.time() - start_time
            print(f"Transcription terminée en {elapsed_time:.1f} secondes")
            
            # Nettoyer le texte transcrit
            text = result["text"].strip()
            
            # Post-traitement
            if text:
                # Supprimer les espaces multiples
                text = ' '.join(text.split())
                # Pour l'arabe: nettoyage spécifique
                if language == 'ar':
                    # Normaliser les caractères arabes
                    text = self._clean_arabic_text(text)
                else:
                    # Capitaliser la première lettre pour les langues latines
                    if text and text[0].islower():
                        text = text[0].upper() + text[1:]
                
                # Ajouter un point final si absent
                if text and text[-1] not in ['.', '!', '?', '۔', '؟']:
                    text += '.' if language in ['fr', 'en'] else '۔' if language == 'ar' else '.'
            
            return text if text else "Aucun texte transcrit détecté"
            
        except Exception as e:
            print(f"Erreur lors de la transcription: {e}")
            return f"Erreur de transcription: {str(e)}"
    
    def _clean_arabic_text(self, text: str) -> str:
        """
        Nettoie et normalise le texte arabe
        """
        # Normaliser les lettres arabes
        replacements = {
            'أ': 'ا',
            'إ': 'ا',
            'آ': 'ا',
            'ة': 'ه',
            'ى': 'ي',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Supprimer les caractères non-arabes (sauf ponctuation et chiffres)
        import re
        text = re.sub(r'[^\u0600-\u06FF\s\d\.،؛؟!]', '', text)
        
        # Normaliser les espaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def get_supported_languages(self) -> dict:
        """
        Retourne les langues supportées par Whisper avec leurs codes
        """
        return {
            'fr': 'Français',
            'ar': 'العربية (Arabe)',
            'en': 'English',
            'es': 'Español',
            'de': 'Deutsch',
            'it': 'Italiano',
            'pt': 'Português',
            'ru': 'Русский',
            'zh': '中文',
            'ja': '日本語',
            'ko': '한국어',
            'hi': 'हिन्दी',
            # Ajoutez d'autres langues au besoin
        }
    
    def transcribe_audio_bytes(self, audio_bytes: bytes, file_format: str = "mp3", language: str = 'fr') -> str:
        """
        Transcrit des bytes audio directement
        """
        try:
            # Créer un fichier temporaire
            with tempfile.NamedTemporaryFile(suffix=f".{file_format}", delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name
            
            # Transcrire
            text = self.transcribe_audio_file(tmp_path, language)
            
            # Nettoyer
            try:
                os.unlink(tmp_path)
            except:
                pass  # Ignorer les erreurs de suppression
            
            return text
            
        except Exception as e:
            print(f"Erreur traitement audio bytes: {e}")
            return f"Erreur traitement audio: {str(e)}"
    
    def convert_audio_format(self, input_path: str, output_format: str = "wav") -> Optional[str]:
        """
        Convertit un fichier audio au format WAV (meilleur pour Whisper)
        """
        try:
            # Charger l'audio
            audio = AudioSegment.from_file(input_path)
            
            # Normaliser le volume
            audio = audio.normalize()
            
            # Convertir en mono si stéréo
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Rééchantillonner à 16kHz si nécessaire
            if audio.frame_rate != 16000:
                audio = audio.set_frame_rate(16000)
            
            # Créer fichier temporaire
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_format}") as tmp_file:
                output_path = tmp_file.name
            
            # Exporter
            audio.export(
                output_path, 
                format=output_format,
                parameters=["-ar", "16000", "-ac", "1"]  # 16kHz, mono
            )
            
            return output_path
            
        except Exception as e:
            print(f"Erreur conversion audio: {e}")
            return None
    
    def validate_audio_file(self, file_path: str) -> Tuple[bool, str]:
        """
        Valide qu'un fichier audio est lisible et approprié
        """
        try:
            if not os.path.exists(file_path):
                return False, "Fichier non trouvé"
            
            # Vérifier la taille
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return False, "Fichier vide"
            if file_size > 50 * 1024 * 1024:  # 50MB
                return False, f"Fichier trop volumineux ({file_size/(1024*1024):.1f}MB)"
            
            # Essayer de charger l'audio
            audio = AudioSegment.from_file(file_path)
            duration = len(audio) / 1000.0  # en secondes
            
            if duration < 0.5:
                return False, f"Audio trop court ({duration:.1f}s)"
            if duration > 600:  # 10 minutes
                return False, f"Audio trop long ({duration/60:.1f}min)"
            
            return True, f"Audio valide: {duration:.1f}s, {audio.channels} canaux, {audio.frame_rate}Hz"
            
        except Exception as e:
            return False, f"Erreur validation: {str(e)}"
    
    def get_model_info(self) -> dict:
        """
        Retourne des informations sur le modèle
        """
        if self.model is None:
            return {"status": "non chargé", "available_models": self.available_models}
        
        return {
            "status": "chargé",
            "model_size": "inconnue",
            "available_models": self.available_models,
            "device": "CPU",  # Whisper utilise CPU par défaut
            "multilingual": True,
            "languages_supported": self.get_supported_languages()
        }