import streamlit as st
import os
import re
import joblib
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
class Config:
    EMOTIONS = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
    MODEL_DIR = 'models_folder'

# --- CORE UTILITIES (From your original code) ---
class TextPreprocessor:
    @staticmethod
    def remove_emojis(text):
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub(r'', text)

    @staticmethod
    def preprocess_text(text, remove_emojis_flag=True):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        if remove_emojis_flag:
            text = TextPreprocessor.remove_emojis(text)
        return text

class ModelComponents:
    def __init__(self):
        self.vectorizer = None
        self.label_encoder = None
        self.are_fitted = False

    def load(self, directory):
        try:
            vectorizer_path = os.path.join(directory, 'fitted_vectorizer.joblib')
            encoder_path = os.path.join(directory, 'fitted_label_encoder.joblib')

            if not os.path.exists(vectorizer_path) or not os.path.exists(encoder_path):
                vectorizer_path = os.path.join(directory, 'custom_vectorizer.joblib')
                encoder_path = os.path.join(directory, 'label_encoder.joblib')

            self.vectorizer = joblib.load(vectorizer_path)
            self.label_encoder = joblib.load(encoder_path)
            self.are_fitted = True
            st.success(f"Components loaded successfully from {directory}")
            return True
        except FileNotFoundError:
            st.error(f"Warning: Component files not found in {directory}")
            return False
        except Exception as e:
            st.error(f"Error loading components from {directory}: {e}")
            return False

class ModelLoader:
    @staticmethod
    def load_model_from_dir(model_name, model_dir):
        model_path = os.path.join(model_dir, f'{model_name}_model.joblib')
        try:
            if not os.path.exists(model_path):
                st.error(f"Model file for {model_name} not found at {model_path}")
                return None
            model = joblib.load(model_path)
            st.success(f"Loaded {model_name} model from {model_path}")
            return model
        except Exception as e:
            st.error(f"Error loading {model_name} model: {e}")
            return None

class ModelPredictor:
    @staticmethod
    def predict_emotion(text_input, model_object, components):
        if not components.are_fitted or not components.vectorizer:
            return "Error: Model components not loaded."

        processed_text = TextPreprocessor.preprocess_text(str(text_input))
        if not processed_text:
            return "Error: Empty or invalid text input."

        try:
            vectorized_text = components.vectorizer.transform([processed_text])
            prediction_proba = model_object.predict_proba(vectorized_text)[0]
            predicted_label_index = np.argmax(prediction_proba)

            predicted_emotion = components.label_encoder.inverse_transform([predicted_label_index])[0]
            confidence = prediction_proba[predicted_label_index]

            return {
                'predicted_emotion': predicted_emotion,
                'confidence': float(confidence),
                'probabilities': {
                    emotion: float(prob)
                    for emotion, prob in zip(components.label_encoder.classes_, prediction_proba)
                }
            }
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return f"Error during prediction: {e}"

# --- STREAMLIT APP ---
def load_model_and_components():
    """Load model and components with caching"""
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
        st.session_state.active_model = None
        st.session_state.active_components = None

    if not st.session_state.model_loaded:
        with st.spinner('Loading model and components...'):
            components = ModelComponents()
            if components.load(Config.MODEL_DIR):
                model = ModelLoader.load_model_from_dir('svm', Config.MODEL_DIR)
                if model:
                    st.session_state.active_model = model
                    st.session_state.active_components = components
                    st.session_state.model_loaded = True
                    return True
            st.error("Failed to load model or components. Please check if model files exist.")
            return False
    return True

def get_emotion_color(emotion):
    """Return color for different emotions"""
    color_map = {
        'joy': '#28a745',      # Green
        'love': '#ff69b4',     # Pink
        'surprise': '#ffc107', # Yellow
        'anger': '#dc3545',    # Red
        'sadness': '#6c757d',  # Gray
        'fear': '#fd7e14'      # Orange
    }
    return color_map.get(emotion.lower(), '#007bff')

def get_emotion_emoji(emotion):
    """Return emoji for different emotions"""
    emoji_map = {
        'joy': '😊',
        'love': '❤️',
        'surprise': '😲',
        'anger': '😠',
        'sadness': '😢',
        'fear': '😨'
    }
    return emoji_map.get(emotion.lower(), '🤔')

# --- MAIN STREAMLIT APP ---
def main():
    st.set_page_config(
        page_title="Tweet Emotion Analyzer",
        page_icon="🎭",
        layout="wide"
    )
    
    st.title("🎭 Tweet Emotion Analyzer")
    st.markdown("**Analyze emotions in text using Machine Learning**")
    st.markdown("---")
    
    # Load model
    if not load_model_and_components():
        st.stop()
    
    # Create columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter your text:")
        user_input = st.text_area(
            label="Text to analyze",
            height=150,
            placeholder="Type or paste your text here...",
            label_visibility="collapsed"
        )
        
        analyze_button = st.button("🔍 Analyze Emotion", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("Supported Emotions:")
        for emotion in Config.EMOTIONS:
            emoji = get_emotion_emoji(emotion)
            st.write(f"{emoji} {emotion.title()}")
    
    # Analysis section
    if analyze_button:
        if user_input.strip():
            with st.spinner('Analyzing emotion...'):
                result = ModelPredictor.predict_emotion(
                    user_input, 
                    st.session_state.active_model, 
                    st.session_state.active_components
                )
                
                if isinstance(result, dict):
                    emotion = result['predicted_emotion']
                    confidence = result['confidence']
                    probabilities = result['probabilities']
                    
                    # Display main result
                    emoji = get_emotion_emoji(emotion)
                    color = get_emotion_color(emotion)
                    
                    st.markdown("---")
                    st.subheader("🎯 Prediction Results")
                    
                    # Main prediction
                    st.markdown(
                        f"<h2 style='color: {color}; text-align: center;'>"
                        f"{emoji} {emotion.upper()}</h2>",
                        unsafe_allow_html=True
                    )
                    
                    st.markdown(
                        f"<p style='text-align: center; font-size: 18px;'>"
                        f"Confidence: <strong>{confidence:.1%}</strong></p>",
                        unsafe_allow_html=True
                    )
                    
                    # Probability breakdown
                    st.subheader("📊 Detailed Probabilities")
                    
                    # Sort probabilities for better display
                    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                    
                    for emotion_name, prob in sorted_probs:
                        emoji = get_emotion_emoji(emotion_name)
                        st.write(f"{emoji} **{emotion_name.title()}**: {prob:.1%}")
                        st.progress(prob)
                    
                else:
                    st.error(f"Prediction failed: {result}")
        else:
            st.warning("⚠️ Please enter some text to analyze!")
    
    # Sidebar information
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("This app uses a trained SVM model to classify emotions in text.")
        st.write("**Emotions detected:**")
        for emotion in Config.EMOTIONS:
            emoji = get_emotion_emoji(emotion)
            st.write(f"• {emoji} {emotion.title()}")
        
        st.header("📝 Tips")
        st.write("• Works best with complete sentences")
        st.write("• Try different types of text")
        st.write("• Emojis are automatically removed")
        
        st.header("🔧 Model Info")
        if st.session_state.get('model_loaded', False):
            st.success("✅ Model loaded successfully")
        else:
            st.error("❌ Model not loaded")

if __name__ == "__main__":
    main()