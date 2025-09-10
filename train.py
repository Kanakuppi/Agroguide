import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import pickle
df=pd.read_csv("Crop_recommendation.csv")

x=df.drop('label',axis=1)
y=df['label']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

mx=MinMaxScaler()
x_train=mx.fit_transform(x_train)
x_test=mx.transform(x_test)

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

random_forest=RandomForestClassifier()
random_forest.fit(x_train,y_train)
y_prediction=random_forest.predict(x_test)
score_accuracy=accuracy_score(y_prediction,y_test)


# Create pickle file for StandardScaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(sc, f)

with open('minmax.pkl', 'wb') as f:
    pickle.dump(mx, f)

# Create pickle file for RandomForestClassifier
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(random_forest, f)