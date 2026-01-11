# streamlit_ui.py - Interface utilisateur Streamlit pour MedBot
import streamlit as st
import tempfile
import os
from pathlib import Path
import sys
import time
import json
from datetime import datetime

# Ajouter le dossier courant au chemin Python
sys.path.append(str(Path(__file__).parent))

try:
    from app.whisper_handler import WhisperTranscriber
    from app.audio_processor import AudioProcessor
    from app.nlp_processor import NLPProcessor
    from app.disease_predictor import DiseasePredictor
    from app.response_generator import ResponseGenerator
    from app.severity_manager import SeverityManager
    from app.history_manager import HistoryManager
    from gtts import gTTS
except ImportError as e:
    st.error(f"Erreur d'importation : {e}")
    st.error("Veuillez installer les dépendances avec: pip install -r requirements.txt")
    st.stop()

# Configuration Streamlit
st.set_page_config(
    page_title="MedBot Pro - Assistant Médical IA",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/medbot',
        'Report a bug': "https://github.com/yourusername/medbot/issues",
        'About': "# MedBot Pro\nAssistant médical intelligent avec reconnaissance vocale"
    }
)

# Initialisation état session
if 'whisper_model' not in st.session_state:
    st.session_state.whisper_model = None
if 'audio_text' not in st.session_state:
    st.session_state.audio_text = ""
if 'audio_file_path' not in st.session_state:
    st.session_state.audio_file_path = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'audio_generated' not in st.session_state:
    st.session_state.audio_generated = False
if 'audio_bytes' not in st.session_state:
    st.session_state.audio_bytes = None
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "text"

@st.cache_resource
def load_components():
    """Charge tous les composants"""
    try:
        # Nettoyer les fichiers temporaires
        AudioProcessor.clean_temp_files()
        
        # Charger Whisper (une seule fois)
        st.info("🔄 Chargement du modèle Whisper... (première fois seulement)")
        whisper_model = WhisperTranscriber(model_size="base")
        
        # Charger autres composants
        nlp_processor = NLPProcessor()
        disease_predictor = DiseasePredictor()
        response_generator = ResponseGenerator()
        severity_manager = SeverityManager()
        history_manager = HistoryManager()
        
        st.success("✅ Tous les composants chargés avec succès!")
        return (whisper_model, nlp_processor, disease_predictor, 
                response_generator, severity_manager, history_manager)
    except Exception as e:
        st.error(f"❌ Erreur chargement composants: {e}")
        return None

def transcribe_audio_interface():
    """Interface pour la transcription audio"""
    st.markdown("### 🎙️ **Reconnaissance Vocale**")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Option 1: Upload fichier audio
        uploaded_file = st.file_uploader(
            "**Téléchargez un fichier audio**",
            type=['mp3', 'wav', 'm4a', 'ogg', 'flac', 'aac'],
            help="Formats supportés: MP3, WAV, M4A, OGG, FLAC, AAC. Maximum 50MB."
        )
        
        if uploaded_file is not None:
            with st.spinner("📁 Traitement du fichier audio..."):
                # Sauvegarder le fichier
                audio_path = AudioProcessor.save_uploaded_audio(uploaded_file)
                
                if audio_path:
                    # Afficher informations audio
                    audio_info = AudioProcessor.get_audio_info(audio_path)
                    if audio_info:
                        st.info(f"""
                        **📊 Informations audio:**
                        - Durée: {audio_info['duration_seconds']:.1f} secondes
                        - Format: {uploaded_file.name.split('.')[-1].upper()}
                        - Taille: {audio_info['file_size_mb']:.2f} MB
                        - Canaux: {audio_info['channels']}
                        - Fréquence: {audio_info['sample_rate']} Hz
                        """)
                    
                    st.session_state.audio_file_path = audio_path
    
    with col2:
        st.markdown("### **Actions**")
        
        # Bouton transcription
        transcribe_btn = st.button(
            "🎤 **Transcrire Audio**", 
            type="primary", 
            use_container_width=True,
            disabled=st.session_state.audio_file_path is None,
            key="transcribe_btn"
        )
        
        if transcribe_btn:
            transcribe_audio()
        
        # Bouton effacer
        if st.button("🗑️ Effacer", use_container_width=True, key="clear_audio_btn"):
            st.session_state.audio_text = ""
            st.session_state.audio_file_path = None
            st.rerun()
    
    # Affichage du texte transcrit
    if st.session_state.audio_text:
        st.markdown("### 📝 **Texte Transcrit**")
        
        with st.expander("Voir/Masquer le texte", expanded=True):
            # Zone de texte éditable
            edited_text = st.text_area(
                "Modifiez si nécessaire:",
                value=st.session_state.audio_text,
                height=150,
                key="transcribed_text_editor",
                label_visibility="collapsed"
            )
            
            # Mettre à jour le texte si modifié
            if edited_text != st.session_state.audio_text:
                st.session_state.audio_text = edited_text
                st.rerun()
            
            # Boutons d'action sous le texte
            col_analyze, col_copy = st.columns(2)
            with col_analyze:
                if st.button("🩺 Analyser ce texte", use_container_width=True, key="analyze_transcribed"):
                    if st.session_state.audio_text.strip():
                        st.session_state.current_tab = "text"
                        st.session_state.user_input_text = st.session_state.audio_text
                        st.rerun()
            
            with col_copy:
                if st.button("📋 Copier le texte", use_container_width=True, key="copy_text_btn"):
                    st.code(st.session_state.audio_text, language="text")
                    st.success("✅ Texte copié dans le presse-papier!")

def transcribe_audio():
    """Transcrit l'audio en texte"""
    if not st.session_state.audio_file_path:
        st.warning("⚠️ Veuillez d'abord uploader un fichier audio")
        return
    
    with st.spinner("🎤 Transcription en cours avec Whisper AI..."):
        try:
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Étape 1/3: Chargement du modèle...")
            progress_bar.progress(20)
            
            # Charger le modèle si pas déjà fait
            if st.session_state.whisper_model is None:
                components = load_components()
                if components:
                    st.session_state.whisper_model = components[0]
            
            status_text.text("Étape 2/3: Conversion au format optimal...")
            progress_bar.progress(50)
            
            # Convertir en WAV si nécessaire
            if not st.session_state.audio_file_path.endswith('.wav'):
                wav_path = AudioProcessor.convert_to_wav(st.session_state.audio_file_path)
                if wav_path:
                    # Supprimer l'ancien fichier
                    try:
                        os.unlink(st.session_state.audio_file_path)
                    except:
                        pass
                    st.session_state.audio_file_path = wav_path
            
            status_text.text("Étape 3/3: Transcription...")
            progress_bar.progress(80)
            
            # Transcrire
            text = st.session_state.whisper_model.transcribe_audio_file(
                st.session_state.audio_file_path
            )
            
            progress_bar.progress(100)
            status_text.text("✅ Transcription terminée!")
            
            # Mettre à jour le texte
            st.session_state.audio_text = text
            
            # Supprimer le fichier temporaire
            try:
                if st.session_state.audio_file_path and os.path.exists(st.session_state.audio_file_path):
                    os.unlink(st.session_state.audio_file_path)
            except:
                pass
            st.session_state.audio_file_path = None
            
            # Attendre un peu avant de cacher la barre
            time.sleep(1)
            
        except Exception as e:
            st.error(f"❌ Erreur transcription: {str(e)}")

def process_text_analysis(user_input: str):
    """Traite l'analyse des symptômes à partir du texte"""
    if not user_input.strip():
        st.warning("⚠️ Veuillez décrire vos symptômes")
        return
    
    with st.spinner("🔍 Analyse en cours..."):
        try:
            # Charger les composants
            components = load_components()
            if not components:
                st.error("❌ Impossible de charger les composants d'analyse")
                return
            
            _, nlp_processor, disease_predictor, response_generator, severity_manager, history_manager = components
            
            # Barre de progression
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Étape 1: Analyse NLP
            status_text.text("**Étape 1/5:** Analyse des symptômes...")
            progress_bar.progress(20)
            symptoms = nlp_processor.extract_symptoms(user_input)
            
            # Étape 2: Prédiction maladies
            status_text.text("**Étape 2/5:** Recherche des correspondances...")
            progress_bar.progress(40)
            predictions = disease_predictor.predict_diseases(symptoms)
            
            # Étape 3: Vérification urgence
            status_text.text("**Étape 3/5:** Évaluation de la gravité...")
            progress_bar.progress(60)
            urgency_alert = severity_manager.generate_urgency_alert(symptoms, predictions)
            
            # Étape 4: Génération réponse
            status_text.text("**Étape 4/5:** Préparation des recommandations...")
            progress_bar.progress(80)
            response = response_generator.generate_response(symptoms, predictions, urgency_alert)
            
            # Étape 5: Sauvegarde
            status_text.text("**Étape 5/5:** Sauvegarde de la consultation...")
            progress_bar.progress(95)
            history_manager.save_consultation(user_input, symptoms, predictions, response)
            
            progress_bar.progress(100)
            status_text.text("✅ **Analyse terminée avec succès!**")
            
            # Stocker les résultats dans session_state
            st.session_state.analysis_results = {
                'symptoms': symptoms,
                'predictions': predictions,
                'response': response,
                'urgency_alert': urgency_alert,
                'user_input': user_input
            }
            st.session_state.audio_generated = False
            st.session_state.audio_bytes = None
            
            time.sleep(0.5)  # Petite pause pour l'affichage
            
        except Exception as e:
            st.error(f"❌ Erreur lors de l'analyse : {str(e)}")
            st.info("Vérifiez que tous les modules sont correctement configurés.")

def display_analysis_results():
    """Affiche les résultats de l'analyse"""
    if not st.session_state.analysis_results:
        return
        
    symptoms = st.session_state.analysis_results['symptoms']
    predictions = st.session_state.analysis_results['predictions']
    response = st.session_state.analysis_results['response']
    urgency_alert = st.session_state.analysis_results['urgency_alert']
    user_input = st.session_state.analysis_results.get('user_input', '')
    
    # Section alerte urgence
    if urgency_alert:
        st.markdown("---")
        st.error(urgency_alert)
        st.markdown("---")
    
    # Layout résultats
    st.success("## ✅ **Analyse médicale complétée**")
    
    # Résumé rapide
    with st.expander("📊 **Résumé rapide**", expanded=True):
        col_symp, col_pred = st.columns(2)
        
        with col_symp:
            st.markdown("**Symptômes identifiés:**")
            if symptoms:
                for symptom in symptoms:
                    st.markdown(f"• {symptom}")
            else:
                st.markdown("*Aucun symptôme spécifique identifié*")
        
        with col_pred:
            st.markdown("**Analyses possibles:**")
            if predictions:
                for disease, score in predictions[:3]:  # Top 3 seulement
                    confidence = min(score * 100, 99)
                    st.markdown(f"• **{disease}** ({confidence:.0f}%)")
            else:
                st.markdown("*Aucune correspondance claire*")
    
    st.markdown("---")
    
    # Analyse médicale complète
    st.markdown("## 📋 **Analyse Médicale Complète**")
    
    # Afficher la réponse formatée
    st.markdown(response)
    
    # Boutons d'action
    st.markdown("---")
    st.markdown("### **Actions**")
    
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button("🔊 Écouter l'analyse", use_container_width=True, key="listen_btn"):
            generate_audio_response(response)
    
    with action_col2:
        if st.button("📄 Générer PDF", use_container_width=True, key="pdf_btn"):
            generate_pdf_report(symptoms, predictions, response, urgency_alert, user_input)
    
    with action_col3:
        if st.button("🔄 Nouvelle analyse", use_container_width=True, key="new_btn"):
            st.session_state.analysis_results = None
            st.session_state.audio_generated = False
            st.session_state.audio_bytes = None
            st.rerun()
    
    # Afficher l'audio si généré
    if st.session_state.audio_generated and st.session_state.audio_bytes:
        st.markdown("### **Audio Généré**")
        st.audio(st.session_state.audio_bytes, format="audio/mp3")
        
        # Bouton téléchargement audio
        st.download_button(
            label="💾 Télécharger l'audio",
            data=st.session_state.audio_bytes,
            file_name=f"analyse_medbot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3",
            mime="audio/mp3",
            use_container_width=True,
            key="audio_download_btn"
        )

def generate_audio_response(text: str):
    """Génère un audio à partir du texte"""
    with st.spinner("🔊 Génération de l'audio..."):
        try:
            # Nettoyer le texte pour la synthèse vocale
            import re
            
            # Supprimer les émojis et markdown
            emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags
                "]+", flags=re.UNICODE)
            
            clean_text = emoji_pattern.sub(r'', text)
            clean_text = re.sub(r'\*\*(.*?)\*\*', r'\1', clean_text)
            clean_text = re.sub(r'\*(.*?)\*', r'\1', clean_text)
            clean_text = clean_text.replace('#', '')
            
            # Ajouter des pauses naturelles
            clean_text = re.sub(r'([.!?])\s+', r'\1\n\n', clean_text)
            
            # Générer l'audio
            tts = gTTS(text=clean_text, lang='fr', slow=False)
            
            # Sauvegarder dans un fichier temporaire
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            tts.save(temp_path)
            
            # Lire le fichier
            with open(temp_path, 'rb') as f:
                audio_bytes = f.read()
            
            # Supprimer le fichier temporaire
            try:
                os.unlink(temp_path)
            except:
                pass
            
            # Mettre à jour l'état
            st.session_state.audio_bytes = audio_bytes
            st.session_state.audio_generated = True
            
            st.success("✅ Audio généré avec succès!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Erreur génération audio: {str(e)}")

def generate_pdf_report(symptoms, predictions, response, urgency_alert, user_input):
    """Génère un rapport PDF"""
    with st.spinner("📄 Génération du PDF..."):
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors
            from reportlab.lib.units import inch
            from io import BytesIO
            
            # Créer le buffer PDF
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, 
                                   rightMargin=72, leftMargin=72,
                                   topMargin=72, bottomMargin=72)
            
            styles = getSampleStyleSheet()
            
            # Styles personnalisés
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=16,
                textColor=colors.HexColor('#1f77b4'),
                spaceAfter=20,
                alignment=1  # Centré
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=12,
                textColor=colors.HexColor('#2c3e50'),
                spaceAfter=12,
            )
            
            normal_style = ParagraphStyle(
                'Normal',
                parent=styles['Normal'],
                fontSize=10,
                spaceAfter=6,
            )
            
            warning_style = ParagraphStyle(
                'Warning',
                parent=styles['Normal'],
                fontSize=10,
                textColor=colors.red,
                spaceBefore=20,
                alignment=1
            )
            
            # Contenu du PDF
            content = []
            
            # Titre
            content.append(Paragraph("RAPPORT MÉDICAL MEDBOT PRO", title_style))
            content.append(Paragraph(f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}", styles['Normal']))
            content.append(Spacer(1, 20))
            
            # Section description
            content.append(Paragraph("DESCRIPTION DU PATIENT", heading_style))
            if user_input:
                content.append(Paragraph(user_input[:500], normal_style))
            content.append(Spacer(1, 15))
            
            # Section symptômes
            if symptoms:
                content.append(Paragraph("SYMPTÔMES IDENTIFIÉS", heading_style))
                symptoms_text = "<br/>".join([f"• {symptom.title()}" for symptom in symptoms])
                content.append(Paragraph(symptoms_text, normal_style))
                content.append(Spacer(1, 15))
            
            # Section analyses
            if predictions:
                content.append(Paragraph("ANALYSES POSSIBLES", heading_style))
                
                # Créer un tableau
                data = [['Maladie', 'Confiance (%)', 'Gravité']]
                
                for disease, score in predictions:
                    disease_info = DiseasePredictor().get_disease_info(disease)
                    severity = disease_info.get('severity', 'inconnue')
                    confidence = min(score * 100, 99)
                    
                    data.append([disease.title(), f"{confidence:.0f}%", severity.title()])
                
                table = Table(data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                content.append(table)
                content.append(Spacer(1, 15))
            
            # Section recommandations
            content.append(Paragraph("RECOMMANDATIONS", heading_style))
            
            # Nettoyer le texte pour le PDF
            clean_response = response.replace('**', '').replace('*', '').replace('#', '')
            # Limiter la longueur
            clean_response = clean_response[:2000] + "..." if len(clean_response) > 2000 else clean_response
            
            content.append(Paragraph(clean_response, normal_style))
            content.append(Spacer(1, 15))
            
            # Avertissement
            content.append(Spacer(1, 20))
            content.append(Paragraph("ATTENTION: CE RAPPORT NE REMPLACE PAS UNE CONSULTATION MÉDICALE PROFESSIONNELLE", warning_style))
            content.append(Paragraph("En cas d'urgence, composez le 15 (SAMU) ou le 18 (Pompiers)", warning_style))
            
            # Génération du PDF
            doc.build(content)
            buffer.seek(0)
            
            # Bouton de téléchargement
            st.download_button(
                label="📥 Télécharger le PDF",
                data=buffer.getvalue(),
                file_name=f"rapport_medbot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="pdf_download_btn_final"
            )
            
            st.success("PDF généré avec succès!")
            
        except Exception as e:
            st.error(f"Erreur génération PDF: {str(e)}")

def show_history_tab(history_manager):
    """Affiche l'onglet historique"""
    st.markdown("## **Historique des Consultations**")
    
    # Statistiques
    stats = history_manager.get_statistics()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Consultations totales", stats["total_consultations"])
    with col2:
        if stats["last_consultation"]:
            last_date = datetime.fromisoformat(stats["last_consultation"]).strftime('%d/%m/%Y')
            st.metric("Dernière consultation", last_date)
        else:
            st.metric("Dernière consultation", "Aucune")
    with col3:
        if st.button("🗑️ Effacer historique", use_container_width=True):
            if history_manager.clear_history():
                st.success("✅ Historique effacé!")
                st.rerun()
    
    # Liste des consultations
    st.markdown("---")
    
    recent_consultations = history_manager.get_recent_consultations(20)
    
    if not recent_consultations:
        st.info("📝 Aucune consultation dans l'historique")
    else:
        for consult in recent_consultations:
            with st.expander(f"🔸 Consultation #{consult['id']} - {consult['timestamp'][11:16]} {consult['timestamp'][:10]}", True):
                col_info, col_actions = st.columns([3, 1])
                
                with col_info:
                    # Informations
                    st.caption(f"ID: {consult['id']} | {consult['timestamp'][:19]}")
                    
                    if consult['symptoms_detected']:
                        st.write("**Symptômes:**", ", ".join(consult['symptoms_detected'][:5]))
                        if len(consult['symptoms_detected']) > 5:
                            st.write(f"... et {len(consult['symptoms_detected']) - 5} autre(s)")
                    
                    if consult['predictions']:
                        st.write("**Analyses:**")
                        for pred in consult['predictions']:
                            disease = pred.get('disease', '')
                            confidence = pred.get('confidence', 0) * 100
                            st.write(f"• {disease} ({confidence:.0f}%)")
                    
                    st.write("**Résumé:**", consult['response_preview'])
                
                with col_actions:
                    if st.button("📋 Voir", key=f"view_{consult['id']}", use_container_width=True):
                        st.session_state.analysis_results = {
                            'user_input': consult['user_input'],
                            'symptoms': consult['symptoms_detected'],
                            'predictions': [(p['disease'], p['confidence']) for p in consult['predictions']],
                            'response': consult['response_preview'],
                            'urgency_alert': ""
                        }
                        st.session_state.current_tab = "text"
                        st.rerun()

def show_emergency_section():
    """Affiche la section urgences"""
    st.markdown("---")
    
    with st.expander("🚨 **URGENCES MÉDICALES - NUMÉROS IMPORTANTS**", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🔴 Urgences vitales**")
            st.markdown("""
            - **15** : SAMU (urgence médicale)
            - **18** : Pompiers (accident, incendie)
            - **112** : Numéro européen unique
            - **114** : Urgences sourds/malentendants
            """)
        
        with col2:
            st.markdown("**🟡 Symptômes graves**")
            st.markdown("""
            - Difficulté à respirer
            - Douleur thoracique
            - Perte de conscience
            - Saignement abondant
            - Paralysie soudaine
            - Brûlure grave
            - Crise convulsive
            """)
        
        with col3:
            st.markdown("**🟢 Autres numéros**")
            st.markdown("""
            - **17** : Police
            - **115** : Samu social
            - **119** : Enfance en danger
            - **116 117** : Médecin de garde
            """)

def main():
    """Fonction principale"""
    # CSS personnalisé
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stButton button {
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # En-tête
    st.markdown('<h1 class="main-header">🩺 MedBot Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Assistant médical intelligent avec reconnaissance vocale et analyse IA</p>', unsafe_allow_html=True)
    
    # Charger les composants
    components = load_components()
    if not components:
        st.error("Impossible de charger les composants nécessaires. Vérifiez les dépendances.")
        return
    
    whisper_model, nlp_processor, disease_predictor, response_generator, severity_manager, history_manager = components
    
    # Créer les onglets
    tab1, tab2, tab3 = st.tabs(["📝 **Saisie Texte**", "🎙️ **Reconnaissance Vocale**", "📚 **Historique**"])
    
    with tab1:
        # Interface texte
        st.markdown("### **Description des Symptômes**")
        
        # Zone de texte avec placeholder amélioré
        user_input = st.text_area(
            "**Parlez-moi de ce que vous ressentez :**",
            placeholder="""Exemple: Bonjour, j'ai mal à la gorge depuis hier soir et j'ai de la fièvre à 38°C ce matin. 
Je me sens très fatigué, j'ai des frissons et des courbatures. 
La toux est sèche et je n'ai pas d'appétit depuis deux jours.""",
            height=150,
            key="text_input_main",
            help="Décrivez vos symptômes le plus précisément possible pour une analyse plus précise"
        )
        
        # Boutons d'action
        col_analyze, col_example = st.columns([2, 1])
        
        with col_analyze:
            if st.button("🔍 **Analyser mes symptômes**", 
                        type="primary", 
                        use_container_width=True,
                        key="analyze_btn_main"):
                process_text_analysis(user_input)
        
        with col_example:
            if st.button("📋 Exemple", use_container_width=True):
                st.session_state.text_input_main = """J'ai des nausées depuis ce matin avec des vomissements. 
J'ai aussi la diarrhée et des crampes abdominales. 
Je me sens faible et j'ai un peu de fièvre (37.8°C)."""
                st.rerun()
        
        # Afficher les résultats si disponibles
        if st.session_state.analysis_results:
            display_analysis_results()
    
    with tab2:
        # Interface reconnaissance vocale
        transcribe_audio_interface()
        
        # Si nous avons du texte transcrit et pas encore d'analyse, proposer l'analyse
        if st.session_state.audio_text and not st.session_state.analysis_results:
            st.markdown("---")
            st.markdown("### **Analyser le texte transcrit**")
            
            if st.button("🩺 Analyser maintenant", type="primary", use_container_width=True):
                process_text_analysis(st.session_state.audio_text)
        
        # Afficher les résultats si disponibles
        if st.session_state.analysis_results:
            display_analysis_results()
    
    with tab3:
        # Interface historique
        show_history_tab(history_manager)
    
    # Section urgences (toujours visible)
    show_emergency_section()
    
    # Footer
    st.markdown("---")
    
    col_footer1, col_footer2, col_footer3 = st.columns(3)
    
    with col_footer1:
        st.markdown("**Statistiques**")
        stats = history_manager.get_statistics()
        st.markdown(f"• {stats['total_consultations']} consultations")
        if whisper_model:
            model_info = whisper_model.get_model_info()
            st.markdown(f"• Modèle: {model_info.get('model_size', 'base')}")
    
    with col_footer2:
        st.markdown("**Fonctionnalités**")
        st.markdown("• Reconnaissance vocale")
        st.markdown("• Analyse IA")
        st.markdown("• Recommandations personnalisées")
    
    with col_footer3:
        st.markdown("**Avertissement**")
        st.markdown("Cet assistant ne remplace pas un médecin. En cas d'urgence, appelez le 15.")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <b>MedBot Pro v2.1</b> • Assistant médical IA • Développé avec ❤️ pour votre santé • 
    <i>Cet outil ne remplace pas une consultation médicale professionnelle</i>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()