# 🌍 Language Detection NLP App

A Natural Language Processing (NLP) project that automatically detects the **language of a given text** using machine learning.  
The project includes a trained language detection model and an interactive **Streamlit web application** that allows users to test the model in real time.

---

## 🚀 Project Overview

Language detection is a common NLP task used in search engines, chatbots, translation systems, and content moderation.  
This project uses a supervised machine learning approach to classify text into its corresponding language.

The application can detect languages such as:
- English
- Hindi
- French
- Arabic  
*(and other supported languages from the dataset)*

---

## 📊 Dataset Description

The dataset contains two columns:

- **text** – input text written in a specific language  
- **language** – the target label representing the language of the text  

The dataset is used to train a multi-class classification model.

---

## 🧠 Approach & Methodology

### 1. Data Preprocessing
- Converted text to lowercase
- Removed punctuation, numbers, and extra spaces
- Minimal preprocessing to preserve language-specific patterns

> Note: Stopword removal and stemming were intentionally avoided as they reduce language-identifying features.

---

### 2. Feature Extraction
- Used **TF-IDF Vectorization**
- Applied **character-level n-grams** to capture language-specific character patterns

---

### 3. Model Training
- Trained a machine learning classifier on the processed text
- Evaluated model performance using accuracy and classification metrics

---

### 4. Model Deployment
- Built a **Streamlit web application**
- Integrated the trained model to allow real-time language prediction

---

## 🖥️ Streamlit Application Features

- Simple and user-friendly interface
- Text input for language detection
- Instant prediction results
- Supports multiple languages such as English, Hindi, French, and Arabic

---

## 🛠️ Tech Stack & Libraries

- **Python**
- **Pandas & NumPy** – data handling
- **NLTK** – text preprocessing
- **Scikit-learn** – feature extraction & modeling
- **Matplotlib & Seaborn** – data visualization
- **Streamlit** – web application deployment

---

## 📂 Project Structure

