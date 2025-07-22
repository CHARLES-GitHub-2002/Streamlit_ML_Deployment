import streamlit as st
import pandas as pd
import numpy as np 
import shap
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn import datasets
#import shap 
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor


st.write('''
# Boston House Price Prediction App
This app predicts the **Boston House price**!
''')

st.write('----')

#loading the dataset 
boston=pd.read_csv(r"C:\Users\CHARLES\Desktop\BOSTON_HOUSING\BostonHousing.csv")

X=boston.drop(columns='medv')
y=boston['medv']

#sider
#Header of the specified input parameter 
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    crim=st.sidebar.slider('CRIM',X.crim.min(),X.crim.max(),X.crim.mean())
    zn=st.sidebar.slider('ZN',X.zn.min(),X.zn.max(),X.zn.mean())
    indus=st.sidebar.slider('INDUS',X.indus.min(),X.indus.max(),X.indus.mean())
    chas = st.sidebar.slider('CHAS (bounds river)', int(X.chas.min()), int(X.chas.max()), int(round(X.chas.mean())))
    nox = st.sidebar.slider('NOX', float(X.nox.min()), float(X.nox.max()), float(X.nox.mean()))
    rm=st.sidebar.slider('RM',X.rm.min(),X.rm.max(),X.rm.mean())
    age=st.sidebar.slider('AGE',X.age.min(),X.age.max(),X.age.mean())
    dis=st.sidebar.slider('DIS',X.dis.min(),X.dis.max(),X.dis.mean())
    rad = st.sidebar.selectbox('RAD (access to highways)', sorted(X.rad.unique()))
    tax = st.sidebar.slider('TAX', int(X.tax.min()), int(X.tax.max()), int(round(X.tax.mean())))
    ptratio = st.sidebar.slider('PTRATIO', float(X.ptratio.min()), float(X.ptratio.max()), float(X.ptratio.mean()))
    b = st.sidebar.slider('B', float(X.b.min()), float(X.b.max()), float(X.b.mean()))
    lstat=st.sidebar.slider('ISTAT',X.lstat.min(),X.lstat.max(),X.lstat.mean())

    data={'crim':crim,
         'zn': zn,
         'indus':indus,
         'chas':chas,
         'nox':nox,
         'rm':rm,
         'age':age,
         'dis':dis,
         'rad':rad,
         'tax':tax,
         'ptratio':ptratio,
         'b':b,
         'lstat':lstat}
    features=pd.DataFrame(data,index=[0])
    return features
df=user_input_features()
#main pannel
# print specific input parameters 
st.header('Specified input Parameters')
st.write(df)
st.write('---')

# Bulding regression model 
model=RandomForestRegressor()
model.fit(X,y)
#Applying model to make prediction 
prediction=model.predict(df)


st.header('Prediction of MEDV')
st.write(prediction)
st.write('---')

#explaining the model predictions using SHAP values 

explainer=shap.TreeExplainer(model)
shap_values=explainer.shap_values(X)


st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')

# Create a Matplotlib-compatible SHAP summary plot
shap.summary_plot(shap_values, X, show=False)  # Disable auto-rendering

# Create a figure object
fig = plt.gcf()  # Get the current figure (safe here since we suppressed SHAP's default show)

# Show it in Streamlit
st.pyplot(fig, bbox_inches='tight')
st.write('---')

st.header('Feature Importance (SHAP Bar Plot)')

# Create the SHAP bar plot safely
fig = plt.figure()  # Create a fresh figure to attach the SHAP plot to
shap.plots.bar(shap.Explanation(values=shap_values, data=X, feature_names=X.columns), show=False)

# Show it in Streamlit
st.pyplot(fig)