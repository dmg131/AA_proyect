import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from fairlearn.metrics import (
    MetricFrame, 
    selection_rate, 
    false_positive_rate, 
    false_negative_rate
)
from sklearn.metrics import recall_score

def load_and_clean_compas(path):
    """Carga el dataset COMPAS y aplica los filtros estándar de ProPublica."""
    df = pd.read_csv(path)
    
    # Filtros de calidad de datos de ProPublica
    df = df[
        (df['days_b_screening_arrest'] <= 30) & 
        (df['days_b_screening_arrest'] >= -30) &
        (df['is_recid'] != -1) &
        (df['c_charge_degree'] != 'O')
    ]
    
    features = [
        'sex', 'age', 'race', 
        'juv_fel_count', 'juv_misd_count', 'juv_other_count', 
        'priors_count', 'c_charge_degree'
    ]
    target = 'two_year_recid'
    
    return df[features], df[target]

def obtener_preprocesador():
    """Construye y devuelve el ColumnTransformer configurado."""
    numeric_features = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count']
    categorical_features = ['sex', 'race', 'c_charge_degree']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor

def evaluar_rendimiento(model, X_test, y_test):
    """Calcula y devuelve las métricas de rendimiento predictivo estándar."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    
    print("Rendimiento del Modelo")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}\n")
    print("Reporte de Clasificación:\n", classification_report(y_test, y_pred))
    
    return {"AUC": auc, "Accuracy": acc, "F1": f1}

def evaluar_equidad_multivariable(y_true, y_pred, grupo_sensible):
    """
    Genera una auditoría exhaustiva para cualquier variable o combinación.
    Incluye Precision (PPV) para evaluar Predictive Parity.
    """
    metrics_dict = {
        'Accuracy': accuracy_score,
        'Precision (PPV)': precision_score,    # Clave para Predictive Parity
        'TPR (Recall)': recall_score,          # Clave para Equalized Odds
        'FPR (Falsos Positivos)': false_positive_rate, # Métrica ProPublica
        'FNR (Falsos Negativos)': false_negative_rate,
        'Selection Rate': selection_rate       # Clave para Demographic Parity
    }
    
    mf = MetricFrame(
        metrics=metrics_dict,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=grupo_sensible
    )
    
    return mf
