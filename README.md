Richter's Predictor: Modeling Earthquake Damage

**Asignatura:** Sistemas de Aprendizaje Automatico   
**Competicion:** [DrivenData — Richter's Predictor](https://www.drivendata.org/competitions/57/nepal-earthquake/)  
**Metrica de evaluacion:** micro averaged F1 Score  

---

## Descripcion del problema

El 25 de abril de 2015, un terremoto de magnitud 7.8 Mw con epicentro en Gorkha (Nepal) causo aproximadamente 9.000 muertos y 10.000 millones de dolares en danos. El objetivo de esta actividad es predecir el nivel de dano estructural de cada edificio afectado:

- **Grado 1** — Daño bajo  
- **Grado 2** — Daño moderado  
- **Grado 3** — Destruccion completa  

Los datos fueron recopilados por el Kathmandu Living Labs y la Central Bureau of Statistics de Nepal, y representan uno de los mayores datasets post-catastrofe jamas reunidos.

---

## Estructura del repositorio

```
.
├── ACT3_7_Terremotos.ipynb        # Notebook principal con todo el pipeline
├── data/
│   ├── train_values.csv           # 260.601 filas x 38 caracteristicas
│   ├── train_labels.csv           # Etiquetas damage_grade (1/2/3)
│   └── test_values.csv            # 86.868 filas para submission final
├── outputs/
│   └── submission_*.csv           # Ficheros de submission generados
└── README.md
```

> Los ficheros CSV no se incluyen en el repositorio por su tamano. Descargarlos desde la pagina oficial de la competicion: https://www.drivendata.org/competitions/57/nepal-earthquake/data/

---

## Pipeline

### 1. Importacion de datos
Los datos se cargan desde Google Drive montado en Colab (`/content/drive/MyDrive/Terremotos/`). Se verifica la integridad dimensional de cada CSV tras la carga.

### 2. Exploracion y analisis (EDA)
Analisis de distribuciones por grado de dano, graficos de barras apiladas para variables categoricas y mapa de calor de correlaciones entre variables numericas y el target.

### 3. Preprocesamiento y muestreo estratificado
Con 260.601 filas el entrenamiento de SVC seria inviable en Colab. Se aplica **muestreo estratificado proporcional (N=20.000, random_state=42)**, que preserva la distribucion de clases con una desviacion inferior a 0.1 puntos porcentuales respecto al dataset completo.

Pasos de preprocesamiento:
- Label Encoding para variables categoricas
- Imputacion por mediana (SimpleImputer)
- Normalizacion Z-score (StandardScaler)

### 4. Seleccion de caracteristicas — herramientas graficas
Se emplean tres metodos complementarios sin dendrogramas:

| Metodo | Tipo |
|--------|------|
| ANOVA F-statistic (`f_classif`) | Estadistico — diferencias de medias entre clases |
| Mutual Information | Informacion — dependencias no lineales |
| RF Feature Importance (Gini) | Basado en modelo — interacciones entre variables |

Se seleccionan por **consenso** las caracteristicas presentes en el top-15 de al menos 2 de los 3 metodos.

### 5. Seleccion de caracteristicas — dendrogramas
Clustering jerarquico con distancia `1 - |correlacion|` y linkage Ward. Corte al 40% de la distancia maxima. Se elige el representante de mayor importancia RF dentro de cada cluster.

Nota tecnica: se aplica `.fillna(0)` sobre la matriz de correlacion para gestionar columnas con varianza cero en la muestra (correlacion indefinida).

### 6. Division train / validation / test
- Train: 64 %
- Validation: 16 % (holdout)
- Test interno: 20 % (reservado para evaluacion final)
- Cross Validation: StratifiedKFold con n_splits=3

### 7. LazyPredict — ranking automatico
Se utiliza `LazyClassifier` con `custom_metric=micro_f1` para alinear el ranking con la metrica oficial de DrivenData. Entrena y evalua mas de 30 clasificadores sin ajuste de hiperparametros.

```python
from lazypredict.Supervised import LazyClassifier
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=micro_f1)
models, predictions = clf.fit(X_train, X_val, y_train, y_val)
```

### 8. Tres modelos de arboles con Cross Validation

| Modelo | Hiperparametros clave |
|--------|-----------------------|
| DecisionTreeClassifier | max_depth=15, min_samples_split=20, class_weight='balanced' |
| RandomForestClassifier | n_estimators=100, max_depth=20, class_weight='balanced_subsample' |
| GradientBoostingClassifier | n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8 |

Todos se evaluan con `f1_score(average='micro')` en StratifiedKFold 5-fold.

### 9. Modelo SVM con GridSearch y RandomSearch

```python
# GridSearchCV — grilla fija
GridSearchCV(SVC(...), {'C': [1, 10], 'gamma': ['scale'], 'kernel': ['rbf']},
             cv=cv, scoring='f1_micro')

# RandomizedSearchCV — distribucion continua
RandomizedSearchCV(SVC(...),
                   {'C': loguniform(0.1, 100), 'gamma': loguniform(1e-4, 1e-1)},
                   n_iter=10, cv=cv, scoring='f1_micro', random_state=42)
```

### 10. Comparativa final y submission
Se selecciona el modelo con mayor CV micro F1 Score. Se reentrana con train+validation, se evalua en test interno y se genera el CSV de submission para DrivenData.

---

## Dependencias

```
Python >= 3.8
scikit-learn
pandas
numpy
matplotlib
seaborn
scipy
lazypredict >= 0.2.16
xgboost
lightgbm
```

Instalacion rapida en Colab:

```bash
pip install lazypredict xgboost lightgbm --quiet
```

---

## Metrica de evaluacion

La competicion utiliza el **micro averaged F1 Score**, apropiado para clasificacion multiclase con clases desbalanceadas. Calcula TP, FP y FN globalmente sumando todas las clases antes de aplicar la formula del F1.

```python
from sklearn.metrics import f1_score
score = f1_score(y_true, y_pred, average='micro')
```

Referencia: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

---

## Ejecucion en Google Colab

1. Crear una cuenta en DrivenData y unirse a la competicion (gratuito).
2. Descargar los cuatro ficheros CSV de la pagina de datos de la competicion.
3. Subirlos a Google Drive en `MyDrive/Terremotos/`.
4. Abrir el notebook en Colab y ejecutar las celdas en orden.

---

## Resultados

| Modelo | CV micro F1 (5-fold) |
|--------|----------------------|
| Random Forest | mejor resultado |
| Gradient Boosting | segundo |
| SVM (RandomSearch) | tercero |
| SVM (GridSearch) | cuarto |
| Decision Tree | baseline |


---

## Licencia

Uso academico. Dataset bajo los terminos de DrivenData. Codigo bajo licencia MIT.
