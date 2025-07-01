import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

import streamlit as st
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import sys
import os

# Add parent directory to sys.path to import preprocessing.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from streamlit_app.preprocessing import preprocess_text, text_length_features # Import text_length_features

# --- Load Models and Transformers ---
# Define the base path for models
models_path = os.path.join(os.path.dirname(__file__), 'models')

# Load transformers
try:
    tfidf_word_vectorizer = joblib.load(os.path.join(models_path, 'tfidf_word_vectorizer.pkl'))
    tfidf_char_vectorizer = joblib.load(os.path.join(models_path, 'tfidf_char_vectorizer.pkl'))
    length_transformer = joblib.load(os.path.join(models_path, 'length_transformer.pkl'))
except FileNotFoundError:
    st.error("Model atau transformer tidak ditemukan. Pastikan file .pkl ada di direktori 'streamlit_app/models'.")
    st.stop()

# Load models
models = {
    "SVM Split 80:20": joblib.load(os.path.join(models_path, 'best_svm_model_8020.pkl')),
    "SVM Split 70:30": joblib.load(os.path.join(models_path, 'best_svm_model_7030.pkl')),
    "SVM Split 60:40": joblib.load(os.path.join(models_path, 'best_svm_model_6040.pkl')),
    "XGBoost Split 80:20": joblib.load(os.path.join(models_path, 'best_xgb_model_8020.pkl')),
    "XGBoost Split 70:30": joblib.load(os.path.join(models_path, 'best_xgb_model_7030.pkl')),
    "XGBoost Split 60:40": joblib.load(os.path.join(models_path, 'best_xgb_model_6040.pkl')),
}

# Mapping for labels
label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
reverse_label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
labels = ['negative', 'neutral', 'positive']

# --- Prediction Function ---
def predict_sentiment(text, model_name):
    # Preprocess text
    preprocessed_text = preprocess_text(text)
    
    # Transform text using loaded vectorizers
    text_transformed_word = tfidf_word_vectorizer.transform([preprocessed_text])
    text_transformed_char = tfidf_char_vectorizer.transform([preprocessed_text])
    
    # Calculate text length features using the imported function
    text_len_features = text_length_features([preprocessed_text]) # Pass as list to match FunctionTransformer expectation

    # Combine all features
    combined_features = hstack([
        text_transformed_word,
        text_transformed_char,
        csr_matrix(text_len_features)
    ])
    
    # Make prediction
    model = models[model_name]
    prediction = model.predict(combined_features)[0]
    return label_mapping[prediction]

# --- Evaluation Data (Hardcoded from Notebook) ---
evaluation_data = {
    "80:20": {
        "SVM": {
            "metrics": {
                "Accuracy": 0.75, "Precision": 0.74, "Recall": 0.72, "F1-Score": 0.73
            },
            "confusion_matrix": np.array([[424, 56, 26], [76, 169, 20], [56, 21, 166]])
        },
        "XGBoost": {
            "metrics": {
                "Accuracy": 0.73, "Precision": 0.73, "Recall": 0.69, "F1-Score": 0.70
            },
            "confusion_matrix": np.array([[426, 53, 27], [92, 160, 13], [69, 23, 151]])
        }
    },
    "70:30": {
        "SVM": {
            "metrics": {
                "Accuracy": 0.75, "Precision": 0.75, "Recall": 0.72, "F1-Score": 0.73
            },
            "confusion_matrix": np.array([[646, 80, 32], [115, 256, 27], [91, 34, 239]])
        },
        "XGBoost": {
            "metrics": {
                "Accuracy": 0.73, "Precision": 0.74, "Recall": 0.69, "F1-Score": 0.70
            },
            "confusion_matrix": np.array([[643, 88, 27], [141, 240, 17], [112, 32, 220]])
        }
    },
    "60:40": {
        "SVM": {
            "metrics": {
                "Accuracy": 0.74, "Precision": 0.74, "Recall": 0.71, "F1-Score": 0.72
            },
            "confusion_matrix": np.array([[843, 110, 58], [141, 349, 40], [130, 44, 312]])
        },
        "XGBoost": {
            "metrics": {
                "Accuracy": 0.72, "Precision": 0.72, "Recall": 0.69, "F1-Score": 0.70
            },
            "confusion_matrix": np.array([[840, 117, 54], [169, 335, 26], [140, 58, 288]])
        }
    }
}

# --- Streamlit App Layout ---
st.set_page_config(page_title="Analisis Sentimen Program Makan Bergizi Gratis", layout="wide")

st.title("Aplikasi Analisis Sentimen Program Makan Bergizi Gratis")

# Sidebar for navigation
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Prediksi Sentimen", "Hasil Evaluasi Model"])

if page == "Prediksi Sentimen":
    st.header("Prediksi Sentimen Teks")
    
    text_input = st.text_area("Masukkan teks yang ingin dianalisis sentimennya:", height=150)
    
    model_choice = st.selectbox(
        "Pilih Model:",
        list(models.keys())
    )
    
    if st.button("Prediksi Sentimen"):
        if text_input:
            with st.spinner(f"Menganalisis sentimen menggunakan {model_choice}..."):
                sentiment = predict_sentiment(text_input, model_choice)
                st.success(f"Sentimen: **{sentiment.upper()}**")
        else:
            st.warning("Mohon masukkan teks untuk prediksi.")

elif page == "Hasil Evaluasi Model":
    st.header("Hasil Evaluasi Model")

    st.write("Berikut adalah perbandingan metrik evaluasi dan confusion matrix dari model SVM dan XGBoost dengan berbagai rasio split data, berdasarkan hasil dari notebook Jupyter.")

    split_choice = st.selectbox("Pilih Rasio Split Data:", list(evaluation_data.keys()))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"SVM ({split_choice})")
        svm_eval = evaluation_data[split_choice]["SVM"]
        st.write("### Metrik Klasifikasi")
        metrics_df_svm = pd.DataFrame([svm_eval["metrics"]])
        st.dataframe(metrics_df_svm, hide_index=True)

        st.write("### Confusion Matrix")
        fig_svm, ax_svm = plt.subplots(figsize=(6, 5))
        sns.heatmap(svm_eval["confusion_matrix"], annot=True, fmt='d', cmap='YlOrRd',
                    xticklabels=labels, yticklabels=labels, ax=ax_svm)
        ax_svm.set_title(f'SVM Confusion Matrix ({split_choice})')
        ax_svm.set_xlabel('Predicted Labels')
        ax_svm.set_ylabel('Actual Labels')
        st.pyplot(fig_svm)

        st.write("### Laporan Klasifikasi")
        # Calculate classification report
        # For simplicity, using dummy y_true and y_pred to generate a report-like table
        # In a real scenario, you'd have actual y_true and y_pred from evaluation
        y_true_svm = []
        y_pred_svm = []
        cm_svm = svm_eval["confusion_matrix"]
        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                count = cm_svm[i, j]
                y_true_svm.extend([reverse_label_mapping[true_label]] * count)
                y_pred_svm.extend([reverse_label_mapping[pred_label]] * count)
        
        report_svm = classification_report(y_true_svm, y_pred_svm, target_names=labels, output_dict=True)
        report_df_svm = pd.DataFrame(report_svm).transpose()
        st.dataframe(report_df_svm)

    with col2:
        st.subheader(f"XGBoost ({split_choice})")
        xgb_eval = evaluation_data[split_choice]["XGBoost"]
        st.write("### Metrik Klasifikasi")
        metrics_df_xgb = pd.DataFrame([xgb_eval["metrics"]])
        st.dataframe(metrics_df_xgb, hide_index=True)

        st.write("### Confusion Matrix")
        fig_xgb, ax_xgb = plt.subplots(figsize=(6, 5))
        sns.heatmap(xgb_eval["confusion_matrix"], annot=True, fmt='d', cmap='RdPu',
                    xticklabels=labels, yticklabels=labels, ax=ax_xgb)
        ax_xgb.set_title(f'XGBoost Confusion Matrix ({split_choice})')
        ax_xgb.set_xlabel('Predicted Labels')
        ax_xgb.set_ylabel('Actual Labels')
        st.pyplot(fig_xgb)

        st.write("### Laporan Klasifikasi")
        # Calculate classification report
        y_true_xgb = []
        y_pred_xgb = []
        cm_xgb = xgb_eval["confusion_matrix"]
        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                count = cm_xgb[i, j]
                y_true_xgb.extend([reverse_label_mapping[true_label]] * count)
                y_pred_xgb.extend([reverse_label_mapping[pred_label]] * count)

        report_xgb = classification_report(y_true_xgb, y_pred_xgb, target_names=labels, output_dict=True)
        report_df_xgb = pd.DataFrame(report_xgb).transpose()
        st.dataframe(report_df_xgb)

    st.subheader("Perbandingan Model Berdasarkan Akurasi")
    accuracy_data = []
    for split, models_data in evaluation_data.items():
        for model_type, data in models_data.items():
            accuracy_data.append({
                "Model": f"{model_type} {split}",
                "Accuracy": data["metrics"]["Accuracy"]
            })
    accuracy_df = pd.DataFrame(accuracy_data)
    accuracy_df = accuracy_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    st.dataframe(accuracy_df)

    st.write("### Visualisasi Perbandingan Akurasi")
    fig_accuracy, ax_accuracy = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Accuracy", y="Model", data=accuracy_df, palette="viridis", ax=ax_accuracy)
    ax_accuracy.set_title("Perbandingan Akurasi Model Berdasarkan Rasio Split Data")
    ax_accuracy.set_xlabel("Akurasi")
    ax_accuracy.set_ylabel("Model")
    st.pyplot(fig_accuracy)

