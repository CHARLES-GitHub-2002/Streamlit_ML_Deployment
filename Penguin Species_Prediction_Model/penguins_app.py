import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write(
    """

# Penguin prediction App

This App predicts the **Palmer Penguin** species!

The Data was obtained from the [Data Proffeser Github](https://github.com/dataprofessor?tab=repositories)
"""
)

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://github.com/dataprofessor/code/blob/master/streamlit/part3/penguins_example.csv)
""")

# Collect user input features into a dataframe 
uploaded_file=st.sidebar.file_uploader('Upload your Input CSV file',type=['csv'])
if uploaded_file is not None:
    input_df=pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island=st.sidebar.selectbox('island',('Biscoe','Dream','Torgersen'))
        sex=st.sidebar.selectbox('Sex',('male','female'))
        bill_length_mm=st.sidebar.slider ('Bill Length (mm)',32.1,59.6,43.9)
        bill_depth_mm=st.sidebar.slider('Bill depth (mm)',13.1,21.5,17.2)
        fliper_length_mm=st.sidebar.slider('Fliper_length (mm)',172.0,231.0,201.0)
        body_mass_g=st.sidebar.slider('Body mass (g)',2700.0,63000.0,4207.0)

        data={'island':island,
              'bill_length_mm':bill_length_mm,
              'bill_depth_mm':bill_depth_mm,
              'flipper_length_mm':fliper_length_mm, 
              'body_mass_g':body_mass_g,
              'sex':sex
              }
        features=pd.DataFrame(data,index=[0])
        return features
    input_df=user_input_features()
#combine user input features with entire penguins dataset
# this will be user in the encoding phase 
penguins_raw=pd.read_csv(r"C:\Users\CHARLES\Downloads\penguins_cleaned.csv")
penguins =penguins_raw.drop(columns=['species'])
df=pd.concat([input_df,penguins],axis=0)

#encoding for oridnal features
encode=['sex','island']
for col in encode:
    dummy=pd.get_dummies(df[col],prefix=col)
    df=pd.concat([df,dummy],axis=1)
    del df[col]
df=df[:1] #selects only the first row(the user input data)
 #Display the user input Features 
st.subheader('user input features')
if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameter(shown below) ')
    st.write(df)

# reading the saved model
with open(r"C:\Users\CHARLES\Desktop\models\penguins_clf.pkl", "rb") as f:
    load_clf = pickle.load(f)
# Applying model to make prediction 
prediction =load_clf.predict(df)
prediction_prob=load_clf.predict_proba(df)

st.subheader('prediction')
penguins_species=np.array(['Adelie','Chinstrap','Gentoo'])
st.write(penguins_species[prediction])


st.subheader('Prediction Probability')
st.write(prediction_prob)


st.write('## CHARLES MUNYUA')