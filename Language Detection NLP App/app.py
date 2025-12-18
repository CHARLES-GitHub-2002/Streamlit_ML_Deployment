import pickle
import streamlit as st
import os 
import re
import pandas as pd 


# Load the trained model and vectorizer
with open(r'C:\Users\CHARLES\Desktop\language detection model\language_detection_model.pkl','rb') as f:
    vectorizer,model=pickle.load(f)

st.title('🌍 Language Detection NLP App')  

st.markdown('Enter test in any suppoerted language to detect the language.')

# User input
user_input=st.text_area("Enter test here",height=150)

# prediction button
if st.button("Detect Language"):
    if user_input.strip()=='':  # Check for empty input
        st.warning("Please enter some text to detect the language.")
    else:
        # Preprocess the input text
        def preprocess_test(text):
            text=re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]',' ',text) #remove punctuation
            text=re.sub(r'[0-9]+',' ',text) #remove numbers
            text=text.lower() #convert to lowercase
            return text
        cleaned_text=preprocess_test(user_input)
        
        # Vectorize the input text
        text_vectorized=vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction=model.predict(text_vectorized)
        probability=model.predict_proba(text_vectorized).max()
        
        # Display the result
        st.success(f"Predicted Language: {prediction[0]}")
        st.info(f"Confidence: {probability*100:.2f}%")


