import pandas as pd
penguins=pd.read_csv(r"C:\Users\CHARLES\Downloads\penguins_cleaned.csv")
#ordinal features encoding
df=penguins.copy()
target='species'
encode=['sex','island']

for col in encode:
    dummy=pd.get_dummies(df[col],prefix=col)
    df=pd.concat([df,dummy],axis=1)
    del df[col]


target_mapper={'Adelie':0,'Chinstrap':1,'Gentoo':2}
def target_encode(val):
    return target_mapper[val]
df['species']=df['species'].apply(target_encode)


#Separating X and y

X=df.drop('species',axis=1)
y=df['species']

#Building random forest model 
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier()
clf.fit(X,y)

import pickle

# Example: save to a folder named "models" on your Desktop
pickle.dump(clf, open(r"C:\Users\CHARLES\Desktop\models\penguins_clf.pkl", "wb"))
