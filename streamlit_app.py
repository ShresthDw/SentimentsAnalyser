import streamlit as st
import os
import re
import joblib
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    matthews_corrcoef, cohen_kappa_score, log_loss, roc_curve, auc
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# --- CONFIGURATION ---
class Config:
    # Paths updated to reflect the 'Datasets' folder structure
    MODEL_DIR = 'models_folder'
    NEW_MODEL_DIR = 'newly_trained_models'
    REPORT_DIR = 'model_reports'
    TEST_CSV_PATH = 'Datasets/testing.csv' 
    TRAIN_CSV_PATH = 'Datasets/training.csv'

    EMOTIONS = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
    VOCAB_SIZE = 10000 
    
    FIGSIZE_MEDIUM = (10, 6)

# --- CORE UTILITIES (Refined from Final_06.ipynb) ---

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
        if pd.isna(text) or not isinstance(text, str):
            return ""
        text = text.lower()
        if remove_emojis_flag:
            text = TextPreprocessor.remove_emojis(text)
        return text

class SentimentAnalyzer:
    @staticmethod
    def analyze_sentiment_single(text):
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0: return 'positive'
        elif analysis.sentiment.polarity < 0: return 'negative'
        else: return 'neutral'
        
    @staticmethod
    def analyze_sentiment_batch(texts):
        return [SentimentAnalyzer.analyze_sentiment_single(str(text)) for text in texts]

    @staticmethod
    def add_sentiment_analysis_to_df(df, text_column='text'):
        # st.info("Performing sentiment analysis on text...")
        if text_column not in df.columns:
            st.error(f"Warning: Text column '{text_column}' not found for sentiment analysis.")
            return df
        
        # Preprocess text before feeding to TextBlob
        df['sentiment_processed_text'] = df[text_column].apply(lambda x: TextPreprocessor.preprocess_text(x, remove_emojis_flag=True))
        df['sentiment'] = SentimentAnalyzer.analyze_sentiment_batch(df['sentiment_processed_text'])
        df.drop(columns=['sentiment_processed_text'], inplace=True, errors='ignore')
        # st.success("Sentiment analysis added to DataFrame.")
        return df

class DataLoader:
    @staticmethod
    def _load_and_clean_data(csv_path, dataset_name="dataset"):
        if not os.path.exists(csv_path):
             st.warning(f"Data file '{csv_path}' not found. Cannot perform {dataset_name} operations.")
             return None

        try:
            df = pd.read_csv(csv_path)
            st.info(f"Loaded {len(df)} {dataset_name} samples from '{os.path.basename(csv_path)}'") 
            required_cols = ['text', 'label']
            
            df = df.dropna(subset=required_cols).copy()
            
            if 'emotion' not in df.columns:
                idx_to_emotion = {i: emotion for i, emotion in enumerate(Config.EMOTIONS)}
                if pd.api.types.is_numeric_dtype(df['label']):
                    df['emotion'] = df['label'].map(idx_to_emotion).astype(str)
                else:
                    df['emotion'] = df['label'].astype(str)

            df['emotion'] = df['emotion'].astype(str)
            df = df[df['emotion'].isin(Config.EMOTIONS)]
            
            if df.empty:
                st.error(f"No valid samples remaining in {dataset_name} after cleaning.")
                return None
            
            st.info(f"After cleaning/mapping: {len(df)} valid samples in {dataset_name}")
            return df
        except Exception as e:
            st.error(f"ERROR loading/cleaning {dataset_name} data from '{os.path.basename(csv_path)}': {e}")
            return None

    @staticmethod
    @st.cache_data(show_spinner=False)
    def load_test_data(csv_path=Config.TEST_CSV_PATH):
        return DataLoader._load_and_clean_data(csv_path, "test")

    @staticmethod
    @st.cache_data(show_spinner=False)
    def load_train_data(csv_path=Config.TRAIN_CSV_PATH):
        return DataLoader._load_and_clean_data(csv_path, "training")

class ModelComponents:
    def __init__(self):
        self.vectorizer = None
        self.label_encoder = LabelEncoder()
        self.are_fitted = False
        self.source_dir = None

    def load(self, directory):
        self.source_dir = directory
        
        try:
            vectorizer_path = os.path.join(directory, 'fitted_vectorizer.joblib')
            encoder_path = os.path.join(directory, 'fitted_label_encoder.joblib')
            
            if not os.path.exists(vectorizer_path) or not os.path.exists(encoder_path):
                vectorizer_path = os.path.join(directory, 'custom_vectorizer.joblib')
                encoder_path = os.path.join(directory, 'label_encoder.joblib')

            if not os.path.exists(vectorizer_path) or not os.path.exists(encoder_path):
                 raise FileNotFoundError(f"Component files not found in {os.path.basename(directory)}.")
            
            self.vectorizer = joblib.load(vectorizer_path)
            self.label_encoder = joblib.load(encoder_path)
            self.are_fitted = True
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            st.error(f"Error loading components from '{os.path.basename(directory)}': {e}")
            return False
        
    def fit(self, train_texts, train_labels, source_info):
        processed_train_texts = [TextPreprocessor.preprocess_text(str(text)) for text in train_texts]
        self.vectorizer = TfidfVectorizer(max_features=Config.VOCAB_SIZE)
        self.vectorizer.fit(processed_train_texts)
        self.label_encoder.fit(train_labels)
        self.are_fitted = True
        self.source_dir = source_info

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        if self.vectorizer: joblib.dump(self.vectorizer, f'{directory}/fitted_vectorizer.joblib')
        if hasattr(self.label_encoder, 'classes_') and self.label_encoder.classes_ is not None:
             joblib.dump(self.label_encoder, f'{directory}/fitted_label_encoder.joblib')
        st.success(f"Fitted components saved to '{os.path.basename(directory)}'")


class ModelLoader:
    @staticmethod
    def load_specific_model(model_name, model_dir):
        model_path = os.path.join(model_dir, f'{model_name}_model.joblib')
        
        if model_name == 'logistic_regression':
            fallback_path = os.path.join(model_dir, 'custom_model.joblib')
            if not os.path.exists(model_path) and os.path.exists(fallback_path):
                model_path = fallback_path
        
        try:
            if not os.path.exists(model_path):
                return None
            model = joblib.load(model_path)
            return model
        except Exception:
            return None

    @staticmethod
    def load_models_and_their_components(model_dir_path):
        components_for_this_dir = ModelComponents()
        if not components_for_this_dir.load(model_dir_path):
            pass 

        model_objects = {}
        model_names_to_load = ['svm', 'naive_bayes', 'logistic_regression', 'random_forest']
        for name in model_names_to_load:
            model = ModelLoader.load_specific_model(name, model_dir_path)
            if model: model_objects[name] = model

        return {'model_objects': model_objects, 'loaded_components': components_for_this_dir}

class ModelTrainer:
    def __init__(self, components_to_fit_or_use: ModelComponents, save_dir=Config.NEW_MODEL_DIR):
        self.components = components_to_fit_or_use
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def _train_sklearn_model(self, model, model_name, train_df):
        processed_texts = [TextPreprocessor.preprocess_text(str(text)) for text in train_df['text']]

        if not self.components.are_fitted:
            st.info(f"Fitting components for {model_name}...")
            self.components.fit(train_df['text'], train_df['emotion'], source_info=f"training session in {os.path.basename(self.save_dir)}")
            self.components.save(self.save_dir)

        X_train_vec = self.components.vectorizer.transform(processed_texts)
        y_train_encoded = self.components.label_encoder.transform(train_df['emotion'])

        st.info(f"Training {model_name}...")
        start_time = time.time()
        model.fit(X_train_vec, y_train_encoded)
        st.success(f"{model_name} trained in {time.time() - start_time:.2f}s.")
        
        model_save_path = f'{self.save_dir}/{model_name}_model.joblib'
        joblib.dump(model, model_save_path)
        st.info(f"{model_name} model saved to '{os.path.basename(self.save_dir)}'")
        return model

    def train_naive_bayes(self, train_df):
        return self._train_sklearn_model(MultinomialNB(), 'naive_bayes', train_df)
    def train_logistic_regression(self, train_df):
        return self._train_sklearn_model(LogisticRegression(solver='liblinear', max_iter=1000), 'logistic_regression', train_df)
    def train_svm(self, train_df):
        return self._train_sklearn_model(SVC(probability=True, kernel='linear', random_state=42), 'svm', train_df)
    def train_random_forest(self, train_df):
        return self._train_sklearn_model(RandomForestClassifier(n_estimators=100, random_state=42), 'random_forest', train_df)

    def train_all_models(self, train_df):
        st.subheader("Starting Training for All Models...")
        trained_model_objects = {}
        with st.container():
            trained_model_objects['naive_bayes'] = self.train_naive_bayes(train_df)
            trained_model_objects['logistic_regression'] = self.train_logistic_regression(train_df)
            trained_model_objects['svm'] = self.train_svm(train_df)
            trained_model_objects['random_forest'] = self.train_random_forest(train_df)
        
        st.success("All Scikit-learn models trained and saved.")
        return {'model_objects': trained_model_objects, 'fitted_components': self.components}

class ModelPredictor:
    @staticmethod
    def predict_emotion(text_input, model_name, model_object, components):
        if not components.are_fitted or not components.vectorizer or not hasattr(components.label_encoder, 'classes_'):
            return "Error: Model components not loaded or fitted."

        processed_text = TextPreprocessor.preprocess_text(str(text_input))
        if not processed_text:
            return "Error: Empty or invalid text input."

        try:
            vectorized_text = components.vectorizer.transform([processed_text])
            
            if hasattr(model_object, 'predict_proba') and model_name not in ['svm', 'random_forest'] or (hasattr(model_object, 'probability') and model_object.probability):
                prediction_proba = model_object.predict_proba(vectorized_text)[0]
                predicted_label_index = np.argmax(prediction_proba)
                confidence = prediction_proba[predicted_label_index]
            else:
                predicted_label_index = model_object.predict(vectorized_text)[0]
                prediction_proba = np.zeros(len(components.label_encoder.classes_))
                confidence = 1.0 
                prediction_proba[predicted_label_index] = confidence

            predicted_emotion = components.label_encoder.inverse_transform([predicted_label_index])[0]
            
            probabilities = {
                emotion: float(prob)
                for emotion, prob in zip(components.label_encoder.classes_, prediction_proba)
            }

            return {
                'predicted_emotion': predicted_emotion,
                'confidence': float(confidence),
                'probabilities': probabilities
            }
        except Exception as e:
            st.error(f"Prediction error for {model_name}: {e}")
            return f"Error during prediction: {e}"

# --- EVALUATION AND VISUALIZATION ---

class ModelEvaluator:
    # ... (Keep ModelEvaluator, calculate_all_metrics methods as defined above) ...
    def __init__(self, current_active_models: dict, current_active_components: ModelComponents):
        self.active_models = current_active_models
        self.active_components = current_active_components

    def evaluate_specific_model(self, model_name, test_df):
        if model_name not in self.active_models or self.active_models[model_name] is None:
            return None

        if not self.active_components.are_fitted or not hasattr(self.active_components.label_encoder, 'classes_'):
            return None

        model_object = self.active_models[model_name]
        known_encoder_classes = list(self.active_components.label_encoder.classes_)
        
        test_df_eval = test_df[test_df['emotion'].isin(known_encoder_classes)].copy()
        if test_df_eval.empty:
            return None
        
        test_texts_processed_series = test_df_eval['text'].astype(str).apply(TextPreprocessor.preprocess_text)
        y_true = self.active_components.label_encoder.transform(test_df_eval['emotion'])

        start_time = time.time()
        try:
            X_test = self.active_components.vectorizer.transform(test_texts_processed_series)
            y_pred = model_object.predict(X_test)
            y_proba = model_object.predict_proba(X_test) if hasattr(model_object, "predict_proba") else None
            inference_time = time.time() - start_time

            metrics_calculated = ModelEvaluator.calculate_all_metrics(
                y_true, y_pred, y_proba, known_encoder_classes
            )

            result = {
                'model_name': model_name, 'metrics': metrics_calculated,
                'y_true_encoded': y_true, 'y_pred_encoded': y_pred, 'y_proba': y_proba,
                'inference_time': inference_time, 'class_names': known_encoder_classes,
                'components_source': self.active_components.source_dir
            }
            return result

        except Exception as e:
            st.error(f"Error evaluating {model_name}: {e}")
            return None

    @staticmethod
    def calculate_all_metrics(y_true, y_pred, y_proba, class_names_from_encoder):
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'weighted_precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'weighted_recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
            'cohens_kappa': cohen_kappa_score(y_true, y_pred),
        }
        
        n_enc_classes = len(class_names_from_encoder)
        report_labels = np.arange(n_enc_classes)

        try:
            if len(np.unique(y_true)) > 1 and y_proba is not None and y_proba.shape[1] == n_enc_classes:
                metrics['log_loss'] = log_loss(y_true, y_proba, labels=report_labels)
                
                y_true_bin = label_binarize(y_true, classes=report_labels)
                if y_true_bin.shape[1] == n_enc_classes:
                    metrics['roc_auc'] = roc_auc_score(y_true_bin, y_proba, multi_class='ovr', average='weighted')
                else: metrics['roc_auc'] = None
            else: 
                metrics['log_loss'] = None
                metrics['roc_auc'] = None
        except Exception:
            metrics['log_loss'] = None
            metrics['roc_auc'] = None

        try:
            valid_target_names = [class_names_from_encoder[i] for i in report_labels if i < n_enc_classes]
            metrics['class_report_dict'] = classification_report(
                y_true, y_pred, target_names=valid_target_names, output_dict=True, zero_division=0,
                labels = report_labels[:len(valid_target_names)]
            )
        except Exception:
            metrics['class_report_dict'] = {}
        
        return metrics


class ResultsVisualizer:
    # ... (Keep ResultsVisualizer methods as defined above) ...
    @staticmethod
    def plot_confusion_matrix(y_true_encoded, y_pred_encoded, class_names, model_name):
        if not class_names: return
        labels_for_cm = np.arange(len(class_names))
        cm = confusion_matrix(y_true_encoded, y_pred_encoded, labels=labels_for_cm)

        fig, ax = plt.subplots(figsize=Config.FIGSIZE_MEDIUM)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f'Confusion Matrix - {model_name.upper()}'); ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        st.pyplot(fig)
        plt.close(fig)

    @staticmethod
    def display_detailed_classification_reports(eval_results):
        for model_name, res in eval_results.items():
            report_dict = res.get('metrics', {}).get('class_report_dict')
            if report_dict and isinstance(report_dict, dict):
                report_df = pd.DataFrame(report_dict).transpose().round(4)
                st.subheader(f"{model_name.upper()} Classification Report")
                st.dataframe(report_df)

    @staticmethod
    def plot_metrics_comparison_bar(eval_results):
        if not eval_results: return
        metrics_data = [{'Model': name.upper(), **res.get('metrics', {})} for name, res in eval_results.items()]
        df = pd.DataFrame(metrics_data).set_index('Model')
        key_metrics = ['accuracy', 'weighted_f1', 'macro_f1', 'roc_auc']
        plot_metrics = [m for m in key_metrics if m in df.columns and not df[m].isnull().all()]

        if not plot_metrics: st.warning("No suitable metrics for comparison plot."); return
        
        df_plot = df[plot_metrics]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        df_plot.plot(kind='bar', ax=ax)
        ax.set_title('Model Performance Comparison'); ax.set_ylabel('Score')
        max_y = df_plot.max().max() if not df_plot.empty else 0
        ax.set_ylim(0, max(1.0, (max_y * 1.1) if pd.notnull(max_y) else 1.0))
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)
        plt.close(fig)

    @staticmethod
    def plot_roc_curves(eval_results):
        st.subheader("ROC Curve Analysis (Micro-Average One-vs-Rest)")
        fig, ax = plt.subplots(figsize=Config.FIGSIZE_MEDIUM)
        plotted_any = False
        
        for model_name, result in eval_results.items():
            y_proba = result.get('y_proba')
            y_true_enc = result.get('y_true_encoded')
            class_names = result.get('class_names')
            
            if y_proba is None or y_true_enc is None or not class_names: continue

            num_classes = len(class_names)
            if y_proba.shape[1] != num_classes: continue
            
            report_labels = np.arange(num_classes)
            y_true_bin = label_binarize(y_true_enc, classes=report_labels)
            
            if y_true_bin.shape[1] == 0: continue 

            try:
                fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
                roc_auc_val = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=2, label=f'{model_name.upper()} (AUC = {roc_auc_val:.3f})')
                plotted_any = True
            except Exception: pass

        if plotted_any:
            ax.plot([0,1], [0,1], 'k--', lw=2, label='Chance Level')
            ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate (Micro-Average)'); ax.set_ylabel('True Positive Rate (Micro-Average)')
            ax.set_title('Receiver Operating Characteristic (ROC) - Micro-Average'); ax.legend(loc='lower right'); ax.grid(True)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("No ROC curves were plotted due to missing probability data or insufficient classes.")
            plt.close(fig)

class DatasetAnalyzer:
    # ... (Keep DatasetAnalyzer methods as defined above) ...
    @staticmethod
    def show_dataset_stats(df):
        if df is None: st.warning("No dataset loaded."); return
        st.subheader("Dataset Statistics")
        st.write(f"Total samples: {len(df)}")
        if 'text' in df.columns:
            df['text_length'] = df['text'].astype(str).apply(len)
            st.write("Text length statistics (characters):")
            st.dataframe(df['text_length'].describe().to_frame())
            df.drop(columns=['text_length'], inplace=True, errors='ignore')

    @staticmethod
    def show_emotion_distribution_text(df):
        if df is None or 'emotion' not in df.columns: st.warning("No dataset/emotion column."); return
        st.subheader("Emotion Distribution")
        counts = df['emotion'].value_counts().to_frame("Count")
        percentages = (df['emotion'].value_counts(normalize=True) * 100).to_frame("Percentage (%)")
        st.dataframe(pd.concat([counts, percentages], axis=1).round(2))

    @staticmethod
    def show_sentiment_distribution(df):
        if 'sentiment' not in df.columns:
            st.error("No sentiment analysis available. Run analysis first.")
            return
        
        st.subheader("Sentiment Distribution in Dataset")
        counts = df['sentiment'].value_counts()
        
        fig, ax = plt.subplots(figsize=Config.FIGSIZE_MEDIUM)
        sns.countplot(x='sentiment', data=df, order=counts.index, ax=ax)
        ax.set_title('Sentiment Distribution'); ax.set_ylabel('Count')
        st.pyplot(fig)
        plt.close(fig)

    @staticmethod
    def plot_emotion_distribution_pie(df):
        if df is None or 'emotion' not in df.columns or df['emotion'].empty: st.warning("No data for pie chart."); return
        counts = df['emotion'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
        ax.set_title('Emotion Distribution (Pie Chart)'); ax.axis('equal')
        st.pyplot(fig)
        plt.close(fig)

    @staticmethod
    def plot_emotion_distribution_bar(df):
        if df is None or 'emotion' not in df.columns or df['emotion'].empty: st.warning("No data for bar chart."); return
        fig, ax = plt.subplots(figsize=Config.FIGSIZE_MEDIUM)
        sns.countplot(y='emotion', data=df, order=df['emotion'].value_counts().index, palette='viridis', ax=ax)
        ax.set_title('Emotion Distribution (Bar Chart)'); ax.set_xlabel('Count'); ax.set_ylabel('Emotion')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    @staticmethod
    def plot_emotion_distribution_within_sentiments(df):
        if df is None or not all(c in df.columns for c in ['emotion', 'sentiment']) or df['emotion'].empty or df['sentiment'].empty:
            st.warning("Missing data for emotion distribution within sentiments."); return
        try:
            cross_tab = pd.crosstab(df['sentiment'], df['emotion'], normalize='index').mul(100)
            fig, ax = plt.subplots(figsize=Config.FIGSIZE_MEDIUM)
            cross_tab.plot(kind='bar', stacked=True, colormap='Spectral', ax=ax)
            ax.set_title('Emotion Distribution within Sentiments (%)'); ax.set_ylabel('% Emotions'); ax.set_xlabel('Sentiment')
            ax.tick_params(axis='x', rotation=0)
            ax.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e: st.error(f"Could not generate stacked bar chart: {e}")

    @staticmethod
    def plot_sentiment_emotion_correlation(df):
        if df is None or not all(c in df.columns for c in ['emotion', 'sentiment']) or df['emotion'].empty or df['sentiment'].empty:
            st.warning("Missing data for sentiment-emotion correlation heatmap."); return
        try:
            contingency_table = pd.crosstab(df['sentiment'], df['emotion'])
            fig, ax = plt.subplots(figsize=Config.FIGSIZE_MEDIUM)
            sns.heatmap(contingency_table, annot=True, fmt='d', cmap='coolwarm', ax=ax)
            ax.set_title('Sentiment-Emotion Correlation (Counts)'); ax.set_ylabel('Sentiment'); ax.set_xlabel('Emotion')
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e: st.error(f"Could not generate heatmap: {e}")
        
    @staticmethod
    def determine_best_model(evaluation_results, metric='weighted_f1'):
        if not evaluation_results: return None, None
        best_model_name, best_score = None, -1
        for model_name, result in evaluation_results.items():
            if result and 'metrics' in result and metric in result['metrics'] and result['metrics'][metric] is not None:
                if result['metrics'][metric] > best_score:
                    best_score = result['metrics'][metric]; best_model_name = model_name
        
        if best_model_name:
            st.info(f"Best model based on **{metric}**: **{best_model_name.upper()}** (Score: {best_score:.4f})")
            return best_model_name, evaluation_results[best_model_name]
        st.warning(f"Could not determine best model based on {metric}.")
        return None, None

# --- CACHING AND SESSION STATE ---

@st.cache_resource
def load_initial_models():
    """Initial load of models and components from default directories."""
    st.info(f"Attempting initial load from '{Config.NEW_MODEL_DIR}' and '{Config.MODEL_DIR}'...")
    
    loaded_data = ModelLoader.load_models_and_their_components(Config.NEW_MODEL_DIR)
    
    if not loaded_data['model_objects'] or not loaded_data['loaded_components'].are_fitted:
        loaded_data = ModelLoader.load_models_and_their_components(Config.MODEL_DIR)
        
    if loaded_data['model_objects'] and loaded_data['loaded_components'].are_fitted:
        st.success(f"Initial models & components loaded successfully from: '{os.path.basename(loaded_data['loaded_components'].source_dir)}'")
        return loaded_data['model_objects'], loaded_data['loaded_components']
    else:
        st.warning("No pre-trained models found. Please use the Training tab to train them.")
        return {}, ModelComponents()

# --- MAIN STREAMLIT APP DRIVER ---

def main_streamlit_app():
    st.set_page_config(
        page_title="Twitter Emotion Analyzer - Full Suite",
        page_icon="🎭",
        layout="wide"
    )
    
    st.title("🎭 Twitter Emotion Analysis Full Suite")
    st.caption("Machine Learning Model Training, Evaluation, and Data Analysis")

    # Setup directories
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    os.makedirs(Config.NEW_MODEL_DIR, exist_ok=True)
    os.makedirs(Config.REPORT_DIR, exist_ok=True)
    
    # --- Session State Setup ---
    if 'active_models' not in st.session_state:
        initial_models, initial_components = load_initial_models()
        st.session_state.active_models = initial_models
        st.session_state.active_components = initial_components
        st.session_state.latest_evaluation_results = {}
        st.session_state.test_df = None
        st.session_state.train_df = None
        st.session_state.sentiment_df = {}
    
    # --- Sidebar Status ---
    with st.sidebar:
        st.subheader("Active ML Status")
        if st.session_state.active_components.are_fitted:
            st.success(f"Components Source: {os.path.basename(st.session_state.active_components.source_dir)}")
        else:
            st.error("Components Not Loaded/Fitted")
            
        st.markdown("**Loaded Models:**")
        if st.session_state.active_models:
            for name in st.session_state.active_models.keys():
                st.write(f"- {name.upper()}")
        else:
            st.info("None")
            
        st.markdown("**Data Status:**")
        if st.session_state.test_df is None:
            st.warning(f"Test Data: Needs Load ({Config.TEST_CSV_PATH})")
        else:
            st.success(f"Test Data: Loaded ({len(st.session_state.test_df)} samples)")
        
        if st.session_state.train_df is None:
            st.warning(f"Train Data: Needs Load ({Config.TRAIN_CSV_PATH})")
        else:
            st.success(f"Train Data: Loaded ({len(st.session_state.train_df)} samples)")

    
    # --- Tabs for Functionality ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Text Analysis", 
        "2. Performance Comparison", 
        "3. Dataset Analysis", 
        "4. Model Training", 
        "5. Save Results"
    ])

    with tab1:
        ui_analysis_page()

    with tab2:
        ui_performance_page()

    with tab3:
        ui_dataset_analysis_page()

    with tab4:
        ui_training_page()
        
    with tab5:
        ui_save_results_page()


# --- TAB IMPLEMENTATIONS ---

def ui_analysis_page():
    st.header("1. Text Emotion Analyzer 📝")
    st.markdown("Select a loaded model to predict the emotion of any text input.")
    
    available_models = st.session_state.active_models
    
    if not available_models or not st.session_state.active_components.are_fitted:
        st.error("No models are currently loaded. Please load or train a model first (see 'Model Training' tab).")
        return

    model_options = {name.upper(): name for name in available_models.keys()}
    
    # Determine the 'best model' for default selection
    best_model_name_key = "SVM"
    if st.session_state.latest_evaluation_results:
        best_name, _ = DatasetAnalyzer.determine_best_model(st.session_state.latest_evaluation_results, metric='weighted_f1')
        if best_name and best_name.upper() in model_options:
            best_model_name_key = best_name.upper()
    
    selected_model_key = st.selectbox(
        "Select Model for Analysis:",
        options=list(model_options.keys()),
        index=list(model_options.keys()).index(best_model_name_key) if best_model_name_key in model_options else 0
    )
    
    model_name = model_options[selected_model_key]
    
    st.info(f"Using components from: '{os.path.basename(st.session_state.active_components.source_dir)}'")

    user_input = st.text_area(
        "Enter text to analyze:",
        height=100,
        placeholder="e.g., 'I am so thrilled about this new opportunity!'"
    )
    
    if st.button("🔍 Analyze Emotion", type="primary") and user_input.strip():
        with st.spinner(f'Analyzing with {model_name.upper()}...'):
            result = ModelPredictor.predict_emotion(
                user_input, 
                model_name,
                st.session_state.active_models[model_name], 
                st.session_state.active_components
            )
            
            if isinstance(result, dict):
                emotion = result['predicted_emotion']
                confidence = result['confidence']
                probabilities = result['probabilities']
                
                st.markdown("---")
                col_res, col_prob = st.columns([1, 2])
                
                with col_res:
                    st.subheader("🎯 Prediction Result")
                    emoji = get_emotion_emoji(emotion)
                    color = get_emotion_color(emotion)
                    st.markdown(
                        f"<h2 style='color: {color}; text-align: center;'>"
                        f"{emoji} {emotion.upper()}</h2>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"<p style='text-align: center; font-size: 18px;'>"
                        f"Confidence: **{confidence:.1%}**</p>",
                        unsafe_allow_html=True
                    )
                    
                with col_prob:
                    st.subheader("📊 Detailed Probabilities")
                    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                    for emotion_name, prob in sorted_probs:
                        st.write(f"{get_emotion_emoji(emotion_name)} **{emotion_name.title()}**: {prob:.1%}")
                        st.progress(prob)
            else:
                st.error(f"Prediction failed: {result}")
    elif st.button("🔍 Analyze Emotion") and not user_input.strip():
        st.warning("⚠️ Please enter some text to analyze!")

def ui_performance_page():
    st.header("2. Performance Comparison of Models 📊")
    
    if st.session_state.test_df is None:
        st.session_state.test_df = DataLoader.load_test_data(Config.TEST_CSV_PATH)

    if st.session_state.test_df is None:
        st.error("Test dataset not loaded. Cannot run evaluation.")
        return

    if not st.session_state.active_models or not st.session_state.active_components.are_fitted:
        st.error("No active models or components. Please load/train models first.")
        return
    
    if st.button("🔄 Run Full Model Evaluation", type="primary"):
        evaluator = ModelEvaluator(st.session_state.active_models, st.session_state.active_components)
        results = evaluator.evaluate_all_active_models(st.session_state.test_df)
        st.session_state.latest_evaluation_results = results
        st.rerun() # Rerun to ensure fresh metrics display
    
    eval_results = st.session_state.latest_evaluation_results
    
    if not eval_results:
        st.info("Evaluation results are not available. Click 'Run Full Model Evaluation' above.")
        return
    
    st.markdown("---")
    
    st.subheader("Evaluation Summary")
    summary_data = []
    for name, res in eval_results.items():
        metrics = res['metrics']
        summary_data.append({
            'Model': name.upper(),
            'Accuracy': f"{metrics.get('accuracy', float('nan')):.4f}",
            'Weighted F1': f"{metrics.get('weighted_f1', float('nan')):.4f}",
            'Macro F1': f"{metrics.get('macro_f1', float('nan')):.4f}",
            'ROC AUC': f"{metrics.get('roc_auc', float('nan')):.4f}",
            'Inference Time (s)': f"{res.get('inference_time', float('nan')):.4f}"
        })
    st.dataframe(pd.DataFrame(summary_data).set_index('Model'))
    
    st.markdown("---")
    
    tab_cm, tab_metrics, tab_roc, tab_time = st.tabs([
        "Confusion Matrices", "Classification Metrics", "ROC Curves", "Inference Time"
    ])
    
    with tab_cm:
        st.subheader("Confusion Matrix View")
        for name, res in eval_results.items():
            st.caption(f"**{name.upper()}**")
            ResultsVisualizer.plot_confusion_matrix(res['y_true_encoded'], res['y_pred_encoded'], res['class_names'], name)

    with tab_metrics:
        st.subheader("Detailed Classification Reports")
        ResultsVisualizer.display_detailed_classification_reports(eval_results)
        st.markdown("---")
        st.subheader("Metrics Bar Chart")
        ResultsVisualizer.plot_metrics_comparison_bar(eval_results)

    with tab_roc:
        ResultsVisualizer.plot_roc_curves(eval_results)

    with tab_time:
        st.subheader("Inference Time Comparison")
        ResultsVisualizer.plot_inference_time_comparison(eval_results)

def ui_dataset_analysis_page():
    st.header("3. Analyze Dataset 📈")
    
    if st.session_state.test_df is None:
        st.session_state.test_df = DataLoader.load_test_data(Config.TEST_CSV_PATH)
    
    df = st.session_state.test_df
    
    if df is None:
        st.error("Test dataset not loaded. Cannot perform analysis.")
        return
        
    @st.cache_data(show_spinner="Running TextBlob analysis...")
    def run_sentiment_analysis(df_input):
        return SentimentAnalyzer.add_sentiment_analysis_to_df(df_input.copy(), text_column='text')
            
    if Config.TEST_CSV_PATH not in st.session_state.sentiment_df and 'sentiment' not in df.columns:
        st.subheader("Sentiment Pre-analysis")
        if st.button("Run TextBlob Sentiment Analysis (Slow)"):
            st.session_state.sentiment_df[Config.TEST_CSV_PATH] = run_sentiment_analysis(df)
            df = st.session_state.sentiment_df[Config.TEST_CSV_PATH]
            st.success("Sentiment analysis complete and cached.")
        else:
            st.info("Click the button to run sentiment analysis using TextBlob.")
            return

    if Config.TEST_CSV_PATH in st.session_state.sentiment_df:
        df = st.session_state.sentiment_df[Config.TEST_CSV_PATH]

    st.markdown("---")

    tab_stats, tab_emo, tab_sentiment = st.tabs([
        "Statistics & Distribution", "Emotion Plots", "Sentiment Correlation"
    ])

    with tab_stats:
        DatasetAnalyzer.show_dataset_stats(df)
        st.markdown("---")
        DatasetAnalyzer.show_emotion_distribution_text(df)

    with tab_emo:
        st.subheader("Emotion Distribution Plots")
        col_pie, col_bar = st.columns(2)
        with col_pie:
            st.write("Pie Chart")
            ResultsVisualizer.plot_emotion_distribution_pie(df)
        with col_bar:
            st.write("Bar Chart")
            ResultsVisualizer.plot_emotion_distribution_bar(df)

    with tab_sentiment:
        if 'sentiment' not in df.columns:
            st.warning("Sentiment analysis data is not available.")
            return

        st.subheader("Overall Sentiment Distribution")
        DatasetAnalyzer.show_sentiment_distribution(df)
        
        st.markdown("---")
        st.subheader("Emotion Distribution within each Sentiment")
        ResultsVisualizer.plot_emotion_distribution_within_sentiments(df)
        
        st.markdown("---")
        st.subheader("Sentiment-Emotion Correlation Heatmap")
        ResultsVisualizer.plot_sentiment_emotion_correlation(df)


def ui_training_page():
    st.header("4. Model Training Options 🛠️")
    
    if st.session_state.train_df is None:
        st.session_state.train_df = DataLoader.load_train_data(Config.TRAIN_CSV_PATH)
        
    train_df = st.session_state.train_df
    
    if train_df is None:
        st.error("Training dataset not loaded. Cannot train models.")
        return

    training_components = ModelComponents()
    trainer = ModelTrainer(training_components, Config.NEW_MODEL_DIR)

    st.markdown("Training new models saves them to **'newly\_trained\_models'** and updates the **active models**.")
    
    train_option = st.radio(
        "Select Training Option:",
        ["Train All Models (NB, LR, SVM, RF)", "Train Specific Model"],
        key="train_select"
    )

    if train_option == "Train All Models (NB, LR, SVM, RF)":
        if st.button("🚀 Start Training All Models", type="primary"):
            with st.spinner("Training models..."):
                train_results = trainer.train_all_models(train_df)
                st.session_state.active_components = train_results['fitted_components']
                st.session_state.active_models = train_results['model_objects'] 
                st.session_state.latest_evaluation_results = {}

    elif train_option == "Train Specific Model":
        model_map_train = {'Naive Bayes': 'naive_bayes', 'Logistic Regression': 'logistic_regression', 
                           'SVM': 'svm', 'Random Forest': 'random_forest'}
        
        selected_model_name = st.selectbox(
            "Select Model to Train:",
            options=list(model_map_train.keys())
        )
        model_key = model_map_train[selected_model_name]
        
        if st.button(f"🚀 Start Training {selected_model_name}", type="primary"):
            with st.spinner(f"Training {selected_model_name}..."):
                trained_model_obj = None
                if model_key == 'naive_bayes': trained_model_obj = trainer.train_naive_bayes(train_df)
                elif model_key == 'logistic_regression': trained_model_obj = trainer.train_logistic_regression(train_df)
                elif model_key == 'svm': trained_model_obj = trainer.train_svm(train_df)
                elif model_key == 'random_forest': trained_model_obj = trainer.train_random_forest(train_df)
                
                if trained_model_obj:
                    st.session_state.active_components = training_components
                    st.session_state.active_models[model_key] = trained_model_obj
                    st.session_state.latest_evaluation_results = {}

def ui_save_results_page():
    st.header("5. Save Evaluation Results 💾")
    
    if not st.session_state.latest_evaluation_results:
        st.warning("No evaluation results available to save. Run full evaluation in the 'Performance Comparison' tab first.")
        return

    eval_results = st.session_state.latest_evaluation_results
    
    st.info(f"Evaluation results for {len(eval_results)} models are ready to be saved to the **'{Config.REPORT_DIR}'** folder.")
    
    if st.button("Save Evaluation Reports", type="primary"):
        with st.spinner("Saving reports..."):
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            os.makedirs(Config.REPORT_DIR, exist_ok=True)
            
            summary_data = []
            for model_name, result in eval_results.items():
                metrics = result.get('metrics', {})
                cs = result.get('components_source', 'N/A')
                cs_display = os.path.basename(cs)
                summary_data.append({
                    'Model': model_name.upper(), 'Components Source': cs_display,
                    'Accuracy': metrics.get('accuracy'), 'Weighted_F1': metrics.get('weighted_f1'),
                    'Macro_F1': metrics.get('macro_f1'), 'ROC_AUC': metrics.get('roc_auc'),
                    'Matthews_Corr': metrics.get('matthews_corrcoef'), 'Cohens_Kappa': metrics.get('cohens_kappa'),
                    'Inference_Time_s': result.get('inference_time')})

            summary_df = pd.DataFrame(summary_data)
            summary_path = f"{Config.REPORT_DIR}/model_evaluation_summary_{timestamp}.csv"
            summary_df.to_csv(summary_path, index=False)
            st.success(f"Evaluation summary saved to: '{os.path.basename(summary_path)}'")

            for model_name, result in eval_results.items():
                report_dict = result.get('metrics', {}).get('class_report_dict')
                if report_dict:
                    report_filename = f"{model_name}_cl_report_{timestamp}.csv"
                    report_file_path = f"{Config.REPORT_DIR}/{report_filename}"
                    pd.DataFrame(report_dict).transpose().to_csv(report_file_path)
            st.success("Individual classification reports saved.")


# --- DRIVER EXECUTION ---
if __name__ == "__main__":
    main_streamlit_app()