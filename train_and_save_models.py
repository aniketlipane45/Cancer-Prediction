import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# Load Dataset
data = pd.read_csv("data.csv")

# Preprocessing
data['diagnosis'] = data['diagnosis'].astype('category').cat.codes
X = data.drop(columns=['id', 'diagnosis'])
y = data['diagnosis']

# Handling Imbalance with SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Define Model Parameters
rf_params = {'n_estimators': [100, 500, 1000]}
svm_params = {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1]}
xgb_params = {'n_estimators': [100, 500, 1000]}

# Function to Train, Evaluate, and Save Model
def train_and_save_model(model, params, filename):
    grid = GridSearchCV(model, params, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    pickle.dump(best_model, open(filename, 'wb'))

    y_pred = best_model.predict(X_test)
    print(f"Model: {filename}")
    print("Best Parameters:", grid.best_params_)
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("-" * 50)

# Train and Save Models
train_and_save_model(RandomForestClassifier(), rf_params, 'random_forest.pkl')
train_and_save_model(SVC(), svm_params, 'svm.pkl')
train_and_save_model(XGBClassifier(), xgb_params, 'xgboost.pkl')
