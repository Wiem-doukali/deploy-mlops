import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import joblib

np.random.seed(42)
symptoms = np.random.randint(0, 2, (500, 10))
labels = np.random.randint(0, 4, 500)

df = pd.DataFrame(symptoms, columns=['fever', 'cough', 'sore_throat', 'fatigue', 'chills', 'headache', 'nausea', 'shortness_of_breath', 'loss_of_taste', 'muscle_ache'])
df['disease'] = labels

X = df.drop('disease', axis=1)
y = df['disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment('Medical Diagnosis Model')
with mlflow.start_run(run_name='Random Forest V1'):
    mlflow.log_param('n_estimators', 100)
    mlflow.log_param('max_depth', 10)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric('test_accuracy', acc)
    mlflow.log_metric('train_accuracy', accuracy_score(y_train, model.predict(X_train)))
    joblib.dump(model, 'model/medical_diagnosis_model.pkl')
    mlflow.sklearn.log_model(model, 'model')
    print(f'✅ Modèle entraîné: {acc:.4f}')