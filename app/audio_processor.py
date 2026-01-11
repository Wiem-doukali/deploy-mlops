# app/audio_processor.py
import streamlit as st
import tempfile
import os
from typing import Optional, Tuple
import io
from pydub import AudioSegment
import numpy as np

class AudioProcessor:
    @staticmethod
    def save_uploaded_audio(uploaded_file) -> Optional[str]:
        """
        Sauvegarde un fichier audio uploadé et retourne le chemin
        """
        try:
            # Créer un dossier temp s'il n'existe pas
            temp_dir = "temp_audio"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Créer un nom de fichier unique
            import uuid
            file_id = str(uuid.uuid4())[:8]
            original_name = uploaded_file.name
            file_ext = os.path.splitext(original_name)[1]
            
            if not file_ext:
                file_ext = ".mp3"  # extension par défaut
            
            file_name = f"audio_{file_id}{file_ext}"
            file_path = os.path.join(temp_dir, file_name)
            
            # Sauvegarder le fichier
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            return file_path
            
        except Exception as e:
            st.error(f"Erreur sauvegarde audio: {e}")
            return None
    
    @staticmethod
    def convert_to_wav(input_path: str) -> Optional[str]:
        """
        Convertit un fichier audio au format WAV
        """
        try:
            # Charger l'audio
            audio = AudioSegment.from_file(input_path)
            
            # Normaliser
            audio = audio.normalize()
            
            # Convertir en mono si nécessaire
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Rééchantillonner à 16kHz (optimal pour Whisper)
            if audio.frame_rate != 16000:
                audio = audio.set_frame_rate(16000)
            
            # Créer un fichier WAV temporaire
            temp_dir = "temp_audio"
            os.makedirs(temp_dir, exist_ok=True)
            
            import uuid
            wav_path = os.path.join(temp_dir, f"converted_{uuid.uuid4()[:8]}.wav")
            
            # Exporter en WAV
            audio.export(
                wav_path,
                format="wav",
                parameters=["-ar", "16000", "-ac", "1", "-sample_fmt", "s16"]
            )
            
            return wav_path
            
        except Exception as e:
            print(f"Erreur conversion WAV: {e}")
            return None
    
    @staticmethod
    def get_audio_info(file_path: str) -> Optional[dict]:
        """
        Retourne des informations sur le fichier audio
        """
        try:
            audio = AudioSegment.from_file(file_path)
            
            return {
                "duration_seconds": len(audio) / 1000.0,
                "channels": audio.channels,
                "sample_rate": audio.frame_rate,
                "sample_width": audio.sample_width,
                "file_size_mb": os.path.getsize(file_path) / (1024 * 1024)
            }
        except Exception as e:
            print(f"Erreur lecture info audio: {e}")
            return None
    
    @staticmethod
    def clean_temp_files(max_age_hours: int = 24):
        """
        Nettoie les fichiers temporaires anciens
        """
        try:
            temp_dir = "temp_audio"
            if not os.path.exists(temp_dir):
                return
            
            import time
            current_time = time.time()
            
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                
                # Vérifier l'âge du fichier
                file_age = current_time - os.path.getmtime(file_path)
                
                if file_age > max_age_hours * 3600:  # Convertir heures en secondes
                    try:
                        os.remove(file_path)
                        print(f"Fichier temporaire supprimé: {filename}")
                    except:
                        pass  # Ignorer les erreurs de suppression
            
        except Exception as e:
            print(f"Erreur nettoyage fichiers temporaires: {e}")
    
    @staticmethod
    def create_audio_preview(file_path: str, max_duration: int = 30) -> Optional[str]:
        """
        Crée une prévisualisation audio (premières 30 secondes)
        """
        try:
            audio = AudioSegment.from_file(file_path)
            
            # Limiter la durée
            if len(audio) > max_duration * 1000:
                audio = audio[:max_duration * 1000]
            
            # Créer un fichier temporaire pour la prévisualisation
            temp_dir = "temp_audio"
            os.makedirs(temp_dir, exist_ok=True)
            
            import uuid
            preview_path = os.path.join(temp_dir, f"preview_{uuid.uuid4()[:8]}.mp3")
            
            # Exporter
            audio.export(preview_path, format="mp3", bitrate="64k")
            
            return preview_path
            
        except Exception as e:
            print(f"Erreur création prévisualisation: {e}")
            return None