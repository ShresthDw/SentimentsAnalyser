import os
import re
import joblib
from flask import Flask, request, jsonify, render_template
import numpy as np

# --- 1. CONFIGURATION ---
class Config:
    EMOTIONS = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
    MODEL_DIR = 'models_folder'

# --- 2. CORE UTILITIES ---
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
            print(f"Components loaded successfully from {directory}")
            return True
        except FileNotFoundError:
            print(f"Warning: Component files not found in {directory}")
            return False
        except Exception as e:
            print(f"Error loading components from {directory}: {e}")
            return False

class ModelLoader:
    @staticmethod
    def load_model_from_dir(model_name, model_dir):
        model_path = os.path.join(model_dir, f'{model_name}_model.joblib')
        try:
            if not os.path.exists(model_path):
                print(f"Model file for {model_name} not found at {model_path}")
                return None
            model = joblib.load(model_path)
            print(f"Loaded {model_name} model from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading {model_name} model: {e}")
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
            print(f"Prediction error: {e}")
            return f"Error during prediction: {e}"

# --- 3. FLASK APPLICATION SETUP ---
app = Flask(__name__)
active_model = None
active_components = None

def load_best_model_for_api():
    global active_model, active_components

    components = ModelComponents()
    if components.load(Config.MODEL_DIR):
        model = ModelLoader.load_model_from_dir('svm', Config.MODEL_DIR)
        if model:
            active_model = model
            active_components = components
            print("Successfully loaded SVM model and components.")
            return

    print("Failed to load a working model or components. The API will not function.")

load_best_model_for_api()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if active_model is None or active_components is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500

    if not request.json or 'text' not in request.json:
        return jsonify({'error': 'Invalid request format. Must be JSON with "text" field.'}), 400

    text_to_predict = request.json['text']
    prediction_result = ModelPredictor.predict_emotion(text_to_predict, active_model, active_components)

    if isinstance(prediction_result, dict):
        return jsonify(prediction_result), 200
    else:
        return jsonify({'error': prediction_result}), 500