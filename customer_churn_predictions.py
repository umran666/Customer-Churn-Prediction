import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

# Load the data

load_data=pd.read_csv('Customer-churn.csv')

# Data Cleaning

load_data=load_data.drop('customerID',axis=1)

load_data['TotalCharges']=pd.to_numeric(load_data['TotalCharges'],errors='coerce')

load_data=load_data.dropna()

# Data Selection

Y=load_data['Churn']
X=load_data.drop('Churn',axis=1)

X=pd.get_dummies(X,drop_first=True)

# train and test split

X_train,X_test,Y_train,Y_test=train_test_split(
    X,Y,test_size=0.2,random_state=42,stratify=Y
)

# Training

Model=RandomForestClassifier(random_state=42)
Model.fit(X_train,Y_train)

# prediction

Prediction=Model.predict(X_test)
Actual_pred=Y_test.values
Comparision=pd.DataFrame({
    'Prediction':Prediction,
    'Actual_pred':Actual_pred
})

# Metrics

acc=accuracy_score(Y_test,Prediction)
pre=precision_score(Y_test,Prediction,pos_label='Yes')
recall=recall_score(Y_test,Prediction,pos_label='Yes')
f1=f1_score(Y_test,Prediction,pos_label='Yes')
print("Accuracy :", acc)
print("Precision:", pre)
print("Recall   :", recall)
print("F1 Score :", f1)

# Confusion Matrix


ConfusionMatrixDisplay.from_predictions(
    Y_test,Prediction
)
plt.title("Confusion Matrix - RF")
plt.show()
