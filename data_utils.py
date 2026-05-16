import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from fairlearn.metrics import (
    MetricFrame, 
    selection_rate, 
    false_positive_rate, 
    false_negative_rate,
    demographic_parity_difference,
    equalized_odds_difference
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
        'sex', 'age', 'age_cat', 'race', 
        'juv_fel_count', 'juv_misd_count', 'juv_other_count', 
        'priors_count', 'c_charge_degree'
    ]
    target = 'two_year_recid'
    
    return df[features], df[target]

def obtener_preprocesador():
    """Construye y devuelve el ColumnTransformer configurado."""
    numeric_features = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count']
    categorical_features = ['sex', 'age_cat', 'race', 'c_charge_degree']

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

def evaluar_rendimiento(model, X_test_proc, y_test):
    """Calcula y devuelve las métricas de rendimiento predictivo estándar."""
    y_pred = model.predict(X_test_proc)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("--- Rendimiento del Modelo ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}\n")
    print("Reporte de Clasificación:\n", classification_report(y_test, y_pred))
    
    return {"Accuracy": acc, "F1": f1}

def obtener_metricas_equidad(y_true, y_pred, grupo_sensible):
    """Genera el MetricFrame de Fairlearn desglosado por el grupo sensible."""
    metrics_dict = {
        'TPR (Recall)': recall_score,
        'FPR (Falsos Positivos)': false_positive_rate,
        'FNR (Falsos Negativos)': false_negative_rate,
        'Selection Rate': selection_rate,
        'Accuracy': accuracy_score
    }
    
    mf = MetricFrame(
        metrics=metrics_dict,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=grupo_sensible
    )
    return mf