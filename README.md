# Proyecto 1 — Responsible IA (Sección 10) ✨

**Universidad del Valle de Guatemala** — 29 de agosto de 2025
**Autor(es):**

* Andrea Ximena Ramirez Recinos (21874)
* Adrian Ricardo Flores Trujillo (21500)
* Daniel Armando Valdez Reyes (21240)

---

## 🧭 Resumen

Este proyecto entrena y evalúa modelos de clasificación para el dataset **Adult (UCI)** con foco en **confiabilidad** y **mitigación de sesgos**:

* **Re‐pesado simple:** `scale_pos_weight` + `sample_weight=fnlwgt` para manejar desbalance y respetar el diseño muestral.
* **Post‐proceso barato:** **calibración por grupo** (isotónica/Platt) y **umbrales por grupo** (igualar *recall* o *precision*).
* **XGBoost con restricciones:** **monotonicidad** (no decrecer con `education-num`/`hours-per-week`) y **restricción de interacciones** con atributos sensibles.
* **Explicabilidad estilo SHAP:** contribuciones direccionales con `pred_contribs=True` (TreeSHAP nativo de XGBoost).
* **Reporte reproducible en HTML:** métricas, curvas ROC/PR, matriz de confusión, importancias, gráficos SHAP-like, **EDA** (distribuciones de objetivo, numéricas y categóricas) y tablas por grupo. Cada nuevo modelo **se concatena** como una nueva sección en el mismo HTML.

---

## 🗂️ Estructura (sugerida)

```
bias-mitigation/
├─ EDA_Responsible.ipynb
├─ report_modelos.html              
├─ README.md
└─ requirements.txt                 
```

> Si trabajás en **Colab**, podés usar directamente el notebook y las celdas provistas.

---

## 📦 Requisitos

**Python** ≥ 3.10

**Dependencias clave:**

* `numpy`, `pandas`, `matplotlib`
* `scikit-learn` ≥ **1.2**  (necesario para `OneHotEncoder(sparse_output=False)` y `set_output('pandas')`)
* `xgboost` ≥ **2.1** (para pasar `early_stopping_rounds` en el **constructor** del `XGBClassifier`)
* `ucimlrepo` (para cargar el dataset Adult automáticamente)

Instalar:

```bash
pip install -r requirements.txt
```

O, en Colab:

```python
%pip -q install numpy pandas matplotlib scikit-learn>=1.2 xgboost>=2.1 ucimlrepo
```

---

## 🚀 Reproducir el proyecto (paso a paso)

1. **Cargar dataset Adult (UCI)**

```python
from ucimlrepo import fetch_ucirepo
adult = fetch_ucirepo(id=2)
X = adult.data.features
y = adult.data.targets  # binaria: <=50K / >50K
display(adult.variables)  # esquema de variables
```

2. **Imports base** (versión limpia que usamos en el notebook)

```python
# Standard library
import base64, io, os, datetime as dt
from typing import Any, Dict, List, Optional, Tuple

# Third-party
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display

# scikit-learn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)

# XGBoost
import xgboost as xgb
from xgboost import XGBClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
```

3. **EDA** (opcional pero recomendado)
   – Usar las funciones `make_y_hist_b64`, `make_numeric_grid_b64`, `make_categorical_grid_b64` y `export_eda_distributions_section` para **insertar EDA al HTML**.
   – La **tabla de variables** (`adult.variables`) se incorpora con `export_dataset_variables_section`.

4. **Preprocesamiento + Pipeline**

   * `ColumnTransformer` con imputación, *standardization* numérica y `OneHotEncoder` **denso** (`sparse_output=False`).
   * `pre.set_output(transform="pandas")` para que XGBoost reciba nombres de columnas.

5. **Entrenamiento con re‐pesado**

   * `sample_weight = X[fnlwgt]` y `scale_pos_weight` calculado con pesos.
   * **Early stopping**: pasar `early_stopping_rounds` en el **constructor** de `XGBClassifier` (no en `fit`).
   * Silenciar logs con `verbosity=0` y `clf__verbose=False`.

6. **Restricciones (solo XGBoost)**

   * **Monotonicidad**: `education-num` y `hours-per-week` con `+1`.
   * **Interacciones**: agrupar por **nombres** de columnas transformadas para limitar interacciones con sensibles (`sex`, `race`).

7. **Validación, umbral y post-proceso**

   * Selección de **umbral por F1** en validación.
   * **Calibración por grupo** (isotónica/Platt) sobre scores de validación.
   * **Umbrales por grupo** para igualar *recall* (o *precision*) global.

8. **Evaluación final + Reporte HTML**

   * `evaluate_on_test_with_html(...)` genera/concatena secciones en `report_modelos.html`:
     métricas, matriz de confusión, **ROC**, **PR**, **importancias**, **SHAP-like**, **tablas por grupo**.
   * Podés correr múltiples modelos (p.ej., XGB y luego LogReg) y se **agregan** secciones al mismo HTML.

---

## 🧪 Qué vas a ver en `report_modelos.html`

* **Métricas (Test):** Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC.
* **Gráficas:**

  * Matriz de confusión, curvas **ROC** y **Precision–Recall**.
  * **Importancias** (XGBoost) y **Contribuciones tipo SHAP** (magnitud, signo, niveles categóricos).
* **Post-proceso por grupo:** tabla con *precision/recall/F1* por subgrupo (p.ej., `sex`) y umbrales utilizados.
* **EDA:** distribuciones de la variable objetivo, numéricas y categóricas, con notas de confiabilidad.

> Cada corrida de `evaluate_on_test_with_html(...)` **anexa** una sección con el nombre del modelo usado (`model_name`).

---

## 🔗 Recursos rápidos

* **Notebook (.ipynb):** descarga directa desde GitHub Pages.
* **Preview HTML del notebook:** visualización rápida en el sitio.
* **Repositorio:** código fuente y notebooks.

---

## 🧑‍⚖️ Nota sobre el uso de `fnlwgt`

`fnlwgt` es un **peso de diseño muestral**. En este proyecto lo usamos como `sample_weight` para **entrenamiento/validación**, **no** como feature predictiva, para no mezclar el proceso de muestreo con la señal del fenómeno.


