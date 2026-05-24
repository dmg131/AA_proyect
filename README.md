# Evaluación Crítica y Mitigación de Sesgos Raciales sobre el Dataset COMPAS

Trabajo de la asignatura *Aprendizaje Avanzado* (curso 2025–2026). Aborda la predicción del riesgo de reincidencia sobre el dataset **COMPAS** de ProPublica combinando tres ejes: rendimiento predictivo, equidad algorítmica (*fairness*) y explicabilidad (*XAI*).

El pipeline final selecciona empíricamente un **XGBoost** como modelo *baseline* (AUC-ROC = 0.7406), aplica mitigación *in-processing* mediante **Reducción de Gradiente Exponencial** bajo restricciones de *Equalized Odds* y audita el comportamiento resultante con **SHAP** y **LIME** a nivel global y local.

## Resultados principales

- **Selección del baseline:** comparativa con validación cruzada estratificada (5-folds) sobre 1 164 configuraciones de Regresión Logística, Árbol de Decisión, Random Forest, SVM y XGBoost.
- **Mitigación:** reducción del *Equalized Odds Gap* racial del **68.35 %** (0.2812 → 0.0890) frente a una degradación controlada del **1.68 %** en *accuracy* (0.6971 → 0.6803).
- **Efecto interseccional:** las brechas *race × sex* y *race × age*, no incluidas explícitamente en la restricción, también se reducen sustancialmente.
- **Explicabilidad:** SHAP y LIME confirman que las variables dominantes son `priors_count` y `age`, con presencia residual pero auditable de atributos demográficos.

## Estructura del repositorio

```
.
├── dataset/                          # Datasets COMPAS originales y versión limpia
│   ├── compas-scores-two-years.csv
│   ├── compas-scores.csv
│   └── compas_two_years_limpio.csv
├── Images/                           # Figuras generadas para el informe
├── report/                           # Memoria LaTeX (proyecto_main.tex)
├── data_utils.py                     # Carga, limpieza, preprocesador y evaluación
├── eda.ipynb                         # Análisis exploratorio (Sección 3.3)
├── experimentos_y_resultados.ipynb   # Selección del baseline (Sección 4.2 / 5.1)
├── fairness.ipynb                    # Mitigación in-processing (Sección 4.3 / 5.2)
├── main_explicabilidad.ipynb         # Análisis SHAP y LIME (Sección 4.4 / 5.3)
├── requirements.txt
└── README.md
```

## Requisitos

- Python ≥ 3.10
- Dependencias listadas en [`requirements.txt`](requirements.txt): `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `fairlearn`, `shap`, `lime`, `notebook`.

## Instalación

```bash
git clone https://github.com/dmg131/AA_proyect.git
cd AA_proyect
python -m venv .venv
source .venv/bin/activate          # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducir los experimentos

Los *notebooks* están pensados para ejecutarse en el orden en que aparecen en la memoria:

1. **`eda.ipynb`** — limpieza siguiendo los filtros de ProPublica y análisis exploratorio.
2. **`experimentos_y_resultados.ipynb`** — *Grid Search* con `StratifiedKFold(k=5)` sobre los cinco candidatos; selecciona XGBoost como *baseline*.
3. **`fairness.ipynb`** — mitigación con `ExponentiatedGradient` (Fairlearn) bajo restricción `EqualizedOdds`; búsqueda del umbral óptimo de tolerancia ε mediante un *Trade-off Score*.
4. **`main_explicabilidad.ipynb`** — explicaciones globales (SHAP *summary*, *dependence plots*) y locales (SHAP *waterfall* y LIME) sobre el individuo 1505 (falso positivo).

Las funciones reutilizables (carga del CSV, filtros de ProPublica, `ColumnTransformer` de preprocesado y métricas de evaluación) están centralizadas en [`data_utils.py`](data_utils.py).

Todos los experimentos fijan `random_state=42` y la partición *train/test* (70/30) se estratifica conjuntamente por la variable objetivo y la raza para preservar la composición demográfica.

## Memoria

El documento completo (introducción, trabajo relacionado, metodología, resultados, discusión y conclusiones) se encuentra en [`report/proyecto_main.tex`](report/proyecto_main.tex). Para compilarlo:

```bash
cd report
pdflatex proyecto_main.tex && pdflatex proyecto_main.tex
```

## Autores

Trabajo realizado en grupo para la asignatura Aprendizaje Avanzado:

- Carlos Vidal Rodríguez
- David Martínez Gómez
- Pau Mateo Lillo
- Stanislav Gatin

El desglose detallado de contribuciones se encuentra en el apéndice de la memoria.

## Datos y licencia

El dataset COMPAS es de dominio público y procede del repositorio oficial de ProPublica: <https://github.com/propublica/compas-analysis>. Este repositorio se distribuye con fines exclusivamente académicos.
