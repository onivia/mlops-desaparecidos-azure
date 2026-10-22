# Databricks notebook source
# MAGIC %md
# MAGIC # NB04_entrenamiento — Entrenamiento y selección del modelo challenger
# MAGIC
# MAGIC Entrena 5 modelos para predecir si una persona desaparecida será **localizada** y registra el mejor (por **AUC-ROC**, desempate **F1**) como `@challenger`.
# MAGIC
# MAGIC **Parámetros de entrada**
# MAGIC - `iteracion` (default 1) y `job_run_id` (opcional): para nombrar el run y la trazabilidad.
# MAGIC
# MAGIC **Entrada:** tabla Delta `workspace.mlops_desaparecidos.tabla_transformacion`.
# MAGIC
# MAGIC **Función**
# MAGIC - División **determinística** 80/20 por `id_registro` (`crc32(id_registro) % 5 == 0` → prueba). Un registro es siempre de entrenamiento o siempre de prueba en todas las iteraciones: así el champion nunca se evalúa con datos con los que fue entrenado.
# MAGIC - 5 modelos sin la variable `Año`: LogisticRegression, RandomForest, HistGradientBoosting, XGBoost y LightGBM.
# MAGIC - Métricas: AUC, F1, accuracy, precision, recall, AUC-PR, AUC de entrenamiento, brecha de sobreajuste, tiempo.
# MAGIC
# MAGIC **Salida**
# MAGIC - `tabla_baseline` (datos de entrenamiento) y `tabla_test` (datos de prueba), sobrescritas.
# MAGIC - Nueva versión de `modelo_desaparecidos` con alias **`@challenger`**.
# MAGIC - `tabla_resultados_entrenamiento` (historial, se agrega una fila por modelo y por iteración; incluye `job_run_id`, registros procesados y las métricas completas para su consulta vía API).
# MAGIC - Run padre `entrenamiento_iterNN` (con 5 runs hijos) en `MLOps-Desaparecidos/NB04_entrenamiento`.
# MAGIC - Task values: `version_challenger`, `algoritmo_challenger`, `auc_challenger`, `f1_challenger`.

# COMMAND ----------

# MAGIC %pip install -q mlflow xgboost lightgbm

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# CELDA 1: PARÁMETROS DE ENTRADA
# Los valores predeterminados permiten ejecutar manualmente el notebook.
# En Jobs, los parámetros suministrados por la Task sobrescriben estos valores.
dbutils.widgets.text("iteracion", "1", "Iteración")
dbutils.widgets.text("job_run_id", "", "Id de ejecución del job")

valor_iteracion = dbutils.widgets.get("iteracion").strip()
try:
    ITERACION = int(valor_iteracion)
except ValueError as exc:
    raise ValueError(f"Parámetro 'iteracion' inválido: {valor_iteracion!r}. Debe ser un entero positivo.") from exc
if ITERACION <= 0:
    raise ValueError(f"Parámetro 'iteracion' inválido: {ITERACION}. Debe ser mayor que cero.")

# Para la ejecución orquestada, configurar 'job_run_id' en la Task del Job Hijo.
JOB_RUN_ID = dbutils.widgets.get("job_run_id").strip()

# COMMAND ----------

# CELDA 2: IMPORTS Y CONFIGURACIÓN
import json, re, time, warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, precision_score,
                             recall_score, average_precision_score, classification_report,
                             ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay)
import sklearn, xgboost, lightgbm
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from databricks.sdk import WorkspaceClient
from pyspark.sql import functions as F

warnings.filterwarnings("ignore")

# ⚠️ Si su workspace usa otro catálogo (p. ej. "main"), cámbielo aquí
#    y en los demás notebooks del pipeline.
CATALOGO = "workspace"
ESQUEMA = "mlops_desaparecidos"
TABLA_ENTRADA = f"{CATALOGO}.{ESQUEMA}.tabla_transformacion"
TABLA_BASELINE = f"{CATALOGO}.{ESQUEMA}.tabla_baseline"
TABLA_TEST = f"{CATALOGO}.{ESQUEMA}.tabla_test"
TABLA_RESULTADOS = f"{CATALOGO}.{ESQUEMA}.tabla_resultados_entrenamiento"
MODELO = f"{CATALOGO}.{ESQUEMA}.modelo_desaparecidos"
ALIAS_CHALLENGER = "challenger"

TARGET = "localizada"
VARIABLES = ["Ciudad", "Localidad", "Sexo", "RangoEdad", "Escolaridad", "EstadoCivil",
             "Etnia", "PaisNacimiento", "PertenenciaGrupal", "RegistradoPor", "Mes", "Dia"]
VARIABLES_EXCLUIDAS = {
    "Año": "Sesgo temporal: los casos recientes han tenido menos tiempo para resolverse",
    "Fecha": "Ya representada por Mes y Dia (y contiene el año)",
    "Estado": "Es el origen de la variable objetivo (fuga de información)",
    "id_registro / lote / posicion": "Identificadores técnicos",
}
COLUMNAS_TRAZABILIDAD = ["id_registro", "lote", "Año", "Fecha"]

RANDOM_STATE = 42
N_PARTICIONES = 5                 # 1 de cada 5 registros a prueba → 20 %
METRICA_SELECCION = "auc"         # AUC-ROC: no depende del umbral y es robusta al desbalance
METRICA_DESEMPATE = "f1"
MIN_FRECUENCIA_CATEGORIA = 20     # categorías con menos registros → "infrecuentes"
FORMATO_SERIALIZACION = "cloudpickle"   # skops rechaza los árboles de RF/XGBoost/LightGBM
REQUISITOS_MODELO = [f"scikit-learn=={sklearn.__version__}", f"xgboost=={xgboost.__version__}",
                     f"lightgbm=={lightgbm.__version__}"]
FECHA_ENTRENAMIENTO = datetime.now(timezone.utc).isoformat()

# ── Experimento MLflow ────────────────────────────────
def contexto_notebook(nombre_por_defecto):
    ruta, host = None, ""
    try:
        ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
        ruta = ctx.notebookPath().get()
        host = "https://" + ctx.browserHostName().get()
    except Exception:
        pass
    return ruta, (ruta.split("/")[-1] if ruta else nombre_por_defecto), host


RUTA_NOTEBOOK, NOMBRE_NOTEBOOK, HOST = contexto_notebook("NB04_entrenamiento")
USUARIO = spark.sql("SELECT current_user()").first()[0]
CARPETA_EXPERIMENTOS = f"/Users/{USUARIO}/MLOps-Desaparecidos"
WorkspaceClient().workspace.mkdirs(CARPETA_EXPERIMENTOS)

mlflow.set_registry_uri("databricks-uc")
EXPERIMENT_ID = mlflow.set_experiment(f"{CARPETA_EXPERIMENTOS}/{NOMBRE_NOTEBOOK}").experiment_id
mlflow.set_experiment_tag("mlflow.experimentKind", "custom_model_development")
mlflow.end_run()
cliente = MlflowClient()

print(f"Experimento: {CARPETA_EXPERIMENTOS}/{NOMBRE_NOTEBOOK} | iteración {ITERACION}")

# COMMAND ----------

# CELDA 3: CARGA Y DIVISIÓN DETERMINÍSTICA 80/20
assert spark.catalog.tableExists(TABLA_ENTRADA), \
    f"No existe {TABLA_ENTRADA}. Ejecute primero NB03_transformacion."

version_entrada = spark.sql(f"DESCRIBE HISTORY {TABLA_ENTRADA} LIMIT 1").first()["version"]
df = spark.read.option("versionAsOf", version_entrada).table(TABLA_ENTRADA)

faltantes = [c for c in VARIABLES + [TARGET] + COLUMNAS_TRAZABILIDAD if c not in df.columns]
assert not faltantes, f"Columnas no encontradas en {TABLA_ENTRADA}: {faltantes}"

# crc32 del id → la partición de cada registro no cambia entre iteraciones
df = df.withColumn(
    "conjunto",
    F.when(F.pmod(F.crc32(F.col("id_registro").cast("binary")), F.lit(N_PARTICIONES)) == 0, "prueba")
     .otherwise("entrenamiento"))

COLUMNAS_TABLAS = COLUMNAS_TRAZABILIDAD + VARIABLES + [TARGET]
df_baseline = df.filter(F.col("conjunto") == "entrenamiento").select(*COLUMNAS_TABLAS)
df_test = df.filter(F.col("conjunto") == "prueba").select(*COLUMNAS_TABLAS)

pdf_train = df_baseline.toPandas()
pdf_test = df_test.toPandas()


def preparar(pdf):
    X = pdf[VARIABLES].fillna("Sin información").astype(str)
    y = pdf[TARGET].astype(int)
    return X, y


X_train, y_train = preparar(pdf_train)
X_test, y_test = preparar(pdf_test)

# ── Validaciones del split ──
for nombre, y in [("entrenamiento", y_train), ("prueba", y_test)]:
    assert y.nunique() == 2, f"El conjunto de {nombre} tiene una sola clase: no se puede entrenar/evaluar."
assert y_train.value_counts().min() >= 10, "Muy pocos registros en una clase de entrenamiento."

resumen_split = pd.DataFrame({
    "conjunto": ["entrenamiento", "prueba"],
    "registros": [len(X_train), len(X_test)],
    "pct_localizada": [round(y_train.mean() * 100, 2), round(y_test.mean() * 100, 2)],
})
cardinalidad = pd.DataFrame({
    "variable": VARIABLES,
    "categorias": [X_train[v].nunique() for v in VARIABLES],
    "categoria_mas_frecuente": [X_train[v].value_counts().index[0] for v in VARIABLES],
})
constantes = cardinalidad.loc[cardinalidad["categorias"] == 1, "variable"].tolist()
if constantes:
    print(f"⚠️ Variables constantes en esta iteración (no aportan): {constantes}")
display(resumen_split)

# COMMAND ----------

# CELDA 4: GUARDAR tabla_baseline Y tabla_test (SOBRESCRITURA)
def guardar_tabla(df_spark, tabla, comentario):
    (df_spark.write.format("delta").mode("overwrite")
        .option("overwriteSchema", "true").saveAsTable(tabla))
    spark.sql(f"COMMENT ON TABLE {tabla} IS '{comentario}'")
    return spark.sql(f"DESCRIBE HISTORY {tabla} LIMIT 1").first()["version"]


version_baseline = guardar_tabla(
    df_baseline, TABLA_BASELINE,
    f"Datos de entrenamiento (80%) de NB04_entrenamiento, iteración {ITERACION}, a partir de {TABLA_ENTRADA} v{version_entrada}")
version_test = guardar_tabla(
    df_test, TABLA_TEST,
    f"Datos de prueba (20%) de NB04_entrenamiento, iteración {ITERACION}, a partir de {TABLA_ENTRADA} v{version_entrada}")

print(f"✅ {TABLA_BASELINE} v{version_baseline} ({len(X_train)} registros)")
print(f"✅ {TABLA_TEST} v{version_test} ({len(X_test)} registros)")

# COMMAND ----------

# CELDA 5: PREPROCESAMIENTO Y LOS 5 MODELOS
# One-Hot dentro del Pipeline: el modelo registrado recibe los datos
# crudos y aplica la misma transformación al predecir.
def crear_preprocesador():
    return ColumnTransformer(
        transformers=[("categoricas",
                       OneHotEncoder(handle_unknown="infrequent_if_exist",
                                     min_frequency=MIN_FRECUENCIA_CATEGORIA,
                                     sparse_output=False, dtype=np.float32),
                       VARIABLES)],
        remainder="drop")


n_neg, n_pos = int((y_train == 0).sum()), int((y_train == 1).sum())
SCALE_POS_WEIGHT = round(n_neg / n_pos, 4)     # compensa el desbalance en XGBoost

HIPERPARAMETROS = {
    "LogisticRegression": {"C": 1.0, "solver": "lbfgs", "max_iter": 1000,
                           "class_weight": "balanced"},
    "RandomForest": {"n_estimators": 300, "max_depth": 15, "min_samples_leaf": 5,
                     "class_weight": "balanced", "n_jobs": -1},
    "HistGradientBoosting": {"learning_rate": 0.1, "max_iter": 300, "max_depth": 8,
                             "class_weight": "balanced"},
    "XGBoost": {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8,
                "colsample_bytree": 0.8, "scale_pos_weight": SCALE_POS_WEIGHT,
                "tree_method": "hist", "eval_metric": "auc", "n_jobs": -1},
    "LightGBM": {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31,
                 "class_weight": "balanced", "n_jobs": -1, "verbose": -1},
}
CLASES_MODELO = {
    "LogisticRegression": LogisticRegression,
    "RandomForest": RandomForestClassifier,
    "HistGradientBoosting": HistGradientBoostingClassifier,
    "XGBoost": XGBClassifier,
    "LightGBM": LGBMClassifier,
}


def crear_pipeline(nombre):
    modelo = CLASES_MODELO[nombre](**HIPERPARAMETROS[nombre], random_state=RANDOM_STATE)
    return Pipeline([("preprocesamiento", crear_preprocesador()), ("modelo", modelo)])


display(pd.DataFrame([{"modelo": m, "hiperparametros": json.dumps(hp)} for m, hp in HIPERPARAMETROS.items()]))

# COMMAND ----------

# CELDA 6: FUNCIONES DE EVALUACIÓN Y REGISTRO
def calcular_metricas(y_true, y_pred, y_prob):
    return {
        "auc": roc_auc_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "auc_pr": average_precision_score(y_true, y_prob),
    }


def figura(tipo, y_true, valores, titulo):
    fig, ax = plt.subplots(figsize=(5, 4))
    if tipo == "confusion":
        ConfusionMatrixDisplay.from_predictions(y_true, valores, cmap="Blues", ax=ax,
                                                display_labels=["Desaparecida", "Localizada"])
    elif tipo == "roc":
        RocCurveDisplay.from_predictions(y_true, valores, ax=ax)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    else:
        PrecisionRecallDisplay.from_predictions(y_true, valores, ax=ax)
    ax.set_title(titulo, fontweight="bold")
    fig.tight_layout()
    return fig


def importancia_variables(pipeline):
    """Caída del AUC al desordenar cada variable original (comparable entre modelos)."""
    resultado = permutation_importance(pipeline, X_test, y_test, scoring="roc_auc",
                                       n_repeats=3, random_state=RANDOM_STATE, n_jobs=1)
    tabla = (pd.DataFrame({"variable": VARIABLES,
                           "caida_auc_media": resultado.importances_mean,
                           "caida_auc_std": resultado.importances_std})
               .sort_values("caida_auc_media", ascending=False).reset_index(drop=True))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(tabla["variable"], tabla["caida_auc_media"], xerr=tabla["caida_auc_std"], color="#2a6f97")
    ax.invert_yaxis()
    ax.set_xlabel("Caída del AUC al permutar la variable")
    ax.set_title("Importancia de variables (permutación)", fontweight="bold")
    fig.tight_layout()
    return tabla, fig


def nombre_param(texto):
    return re.sub(r"[^\w\-\. /]", "_", str(texto))


def entrenar_y_registrar(nombre):
    """Entrena un modelo dentro de un Pipeline y lo registra como run hijo."""
    pipeline = crear_pipeline(nombre)
    with mlflow.start_run(run_name=f"{nombre}_iter{ITERACION:02d}", experiment_id=EXPERIMENT_ID,
                          nested=True) as run_hijo:
        mlflow.set_tags({"modelo": nombre, "rol": "candidato", "iteracion": str(ITERACION),
                         "etapa_pipeline": "04-entrenamiento", "job_run_id": JOB_RUN_ID})
        mlflow.log_params({**PARAMS_COMUNES, "modelo": nombre})
        mlflow.log_params({f"hp_{nombre_param(k)}": v for k, v in HIPERPARAMETROS[nombre].items()})

        inicio = time.time()
        pipeline.fit(X_train, y_train)
        tiempo = time.time() - inicio

        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = pipeline.predict(X_test)
        metricas = calcular_metricas(y_test, y_pred, y_prob)
        auc_train = roc_auc_score(y_train, pipeline.predict_proba(X_train)[:, 1])
        metricas.update({"auc_train": auc_train,
                         "brecha_auc_train_test": auc_train - metricas["auc"],
                         "tiempo_entrenamiento_s": tiempo})
        mlflow.log_metrics(metricas)

        for tipo, valores, ruta in [("confusion", y_pred, "matriz_confusion"),
                                    ("roc", y_prob, "curva_roc"),
                                    ("pr", y_prob, "curva_precision_recall")]:
            fig = figura(tipo, y_test, valores, f"{nombre} — {ruta.replace('_', ' ')}")
            mlflow.log_figure(fig, f"graficas/{ruta}.png")
            plt.close(fig)

        mlflow.log_dict(classification_report(y_test, y_pred, output_dict=True,
                                              target_names=["desaparecida", "localizada"]),
                        "reportes/classification_report.json")
        tabla_imp, fig_imp = importancia_variables(pipeline)
        mlflow.log_text(tabla_imp.to_csv(index=False), "reportes/importancia_variables.csv")
        mlflow.log_figure(fig_imp, "graficas/importancia_variables.png")
        plt.close(fig_imp)
        mlflow.log_dict(HIPERPARAMETROS[nombre], "hiperparametros.json")

        firma = infer_signature(X_train.head(200), pipeline.predict(X_train.head(200)))
        mlflow.sklearn.log_model(pipeline, name="modelo", signature=firma,
                                 input_example=X_train.head(5),
                                 serialization_format=FORMATO_SERIALIZACION,
                                 extra_pip_requirements=REQUISITOS_MODELO)

    print(f"   {nombre:<22} AUC={metricas['auc']:.4f}  F1={metricas['f1']:.4f}  "
          f"Acc={metricas['accuracy']:.4f}  ({tiempo:.1f}s)")
    return {"pipeline": pipeline, "run_id": run_hijo.info.run_id, "y_prob": y_prob,
            "metricas": metricas, "importancia": tabla_imp}

# COMMAND ----------

# CELDA 7: ENTRENAMIENTO, COMPARACIÓN, SELECCIÓN Y REGISTRO DEL CHALLENGER
# Todo ocurre dentro del run padre "entrenamiento_iterNN".
PARAMS_COMUNES = {
    "iteracion": ITERACION,
    "target": TARGET,
    "variables": ", ".join(VARIABLES),
    "n_variables": len(VARIABLES),
    "variables_excluidas": ", ".join(VARIABLES_EXCLUIDAS),
    "division": f"crc32(id_registro) % {N_PARTICIONES} == 0 → prueba",
    "min_frecuencia_categoria": MIN_FRECUENCIA_CATEGORIA,
    "random_state": RANDOM_STATE,
    "n_train": len(X_train),
    "n_test": len(X_test),
    "tabla_entrada_version": version_entrada,
    "tabla_baseline_version": version_baseline,
    "tabla_test_version": version_test,
}

modelos_entrenados = {}
with mlflow.start_run(run_name=f"entrenamiento_iter{ITERACION:02d}", experiment_id=EXPERIMENT_ID,
                      description=f"Comparación de 5 modelos (iteración {ITERACION}); "
                                  f"selección por {METRICA_SELECCION.upper()}") as run_padre:
    ID_RUN_PADRE = run_padre.info.run_id
    mlflow.set_tags({"etapa_pipeline": "04-entrenamiento", "rol": "comparacion",
                     "iteracion": str(ITERACION), "job_run_id": JOB_RUN_ID,
                     "notebook": RUTA_NOTEBOOK or NOMBRE_NOTEBOOK})
    mlflow.log_params({**PARAMS_COMUNES, "modelos": ", ".join(HIPERPARAMETROS),
                       "metrica_seleccion": METRICA_SELECCION,
                       "metrica_desempate": METRICA_DESEMPATE})
    mlflow.log_text(resumen_split.to_csv(index=False), "datos/resumen_split.csv")
    mlflow.log_text(cardinalidad.to_csv(index=False), "datos/cardinalidad_variables.csv")
    mlflow.log_dict(VARIABLES_EXCLUIDAS, "datos/variables_excluidas.json")
    try:
        for tabla, version, nombre_ds, contexto in [
                (TABLA_ENTRADA, version_entrada, "tabla_transformacion", "entrada"),
                (TABLA_BASELINE, version_baseline, "tabla_baseline", "entrenamiento"),
                (TABLA_TEST, version_test, "tabla_test", "prueba")]:
            mlflow.log_input(mlflow.data.load_delta(table_name=tabla, version=version,
                                                    name=nombre_ds), context=contexto)
    except Exception as e:
        print(f"⚠️ No se pudieron registrar los datasets Delta: {e}")

    # ── 1) Entrenar los 5 modelos (runs hijos) ──
    print("Entrenando modelos...")
    for nombre in HIPERPARAMETROS:
        modelos_entrenados[nombre] = entrenar_y_registrar(nombre)

    # ── 2) Tabla comparativa ──
    df_resultados = (pd.DataFrame([{"modelo": n, **r["metricas"], "run_id": r["run_id"]}
                                   for n, r in modelos_entrenados.items()])
                       .sort_values([METRICA_SELECCION, METRICA_DESEMPATE], ascending=False)
                       .reset_index(drop=True))
    df_resultados.insert(0, "ranking", range(1, len(df_resultados) + 1))

    metricas_graf = ["auc", "f1", "accuracy", "precision", "recall", "auc_pr"]
    fig_barras, ax = plt.subplots(figsize=(12, 5))
    ancho = 0.8 / len(df_resultados)
    x = np.arange(len(metricas_graf))
    for i, fila in df_resultados.iterrows():
        ax.bar(x + i * ancho, [fila[m] for m in metricas_graf], ancho, label=fila["modelo"])
    ax.set_xticks(x + ancho * (len(df_resultados) - 1) / 2)
    ax.set_xticklabels([m.upper() for m in metricas_graf])
    ax.set_ylim(0, 1)
    ax.set_title(f"Métricas por modelo en prueba — iteración {ITERACION}", fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig_barras.tight_layout()

    fig_roc, ax = plt.subplots(figsize=(7, 6))
    for nombre in df_resultados["modelo"]:
        RocCurveDisplay.from_predictions(y_test, modelos_entrenados[nombre]["y_prob"], name=nombre, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_title("Curvas ROC — conjunto de prueba", fontweight="bold")
    fig_roc.tight_layout()
    plt.show()

    mlflow.log_text(df_resultados.to_csv(index=False), "comparacion/tabla_comparativa.csv")
    mlflow.log_figure(fig_barras, "comparacion/metricas_por_modelo.png")
    mlflow.log_figure(fig_roc, "comparacion/curvas_roc.png")

    # ── 3) Selección: mayor AUC (desempate F1) ──
    mejor, segundo = df_resultados.iloc[0], df_resultados.iloc[1]
    NOMBRE_GANADOR = mejor["modelo"]
    ganador = modelos_entrenados[NOMBRE_GANADOR]
    metricas_ganador = {k: float(v) for k, v in ganador["metricas"].items()}
    motivo = (f"{NOMBRE_GANADOR} obtiene el mayor {METRICA_SELECCION.upper()} en prueba "
              f"({mejor[METRICA_SELECCION]:.4f}), {mejor[METRICA_SELECCION] - segundo[METRICA_SELECCION]:+.4f} "
              f"frente al segundo ({segundo['modelo']}, {segundo[METRICA_SELECCION]:.4f}).")
    if metricas_ganador["brecha_auc_train_test"] > 0.10:
        print(f"⚠️ Posible sobreajuste: AUC train - test = {metricas_ganador['brecha_auc_train_test']:.3f}")

    mlflow.set_tags({"mejor_modelo": NOMBRE_GANADOR, "mejor_modelo_run_id": ganador["run_id"],
                     "motivo_seleccion": motivo})
    mlflow.log_metrics({f"mejor_{k}": v for k, v in metricas_ganador.items()})
    mlflow.log_dict({"modelo": NOMBRE_GANADOR, "hiperparametros": HIPERPARAMETROS[NOMBRE_GANADOR],
                     "metricas_test": metricas_ganador, "motivo": motivo, "variables": VARIABLES},
                    "challenger/ficha_modelo.json")
    mlflow.log_text(ganador["importancia"].to_csv(index=False), "challenger/importancia_variables.csv")

    # ── 4) Registro del challenger en Unity Catalog ──
    firma = infer_signature(X_train.head(200), ganador["pipeline"].predict(X_train.head(200)))
    info_modelo = mlflow.sklearn.log_model(
        ganador["pipeline"], name="modelo_challenger", signature=firma,
        input_example=X_train.head(5), serialization_format=FORMATO_SERIALIZACION,
        extra_pip_requirements=REQUISITOS_MODELO, registered_model_name=MODELO)

VERSION_CHALLENGER = str(info_modelo.registered_model_version)
print(f"\n🏆 {motivo}")

# COMMAND ----------

# CELDA 8: METADATOS DEL CHALLENGER Y ALIAS @challenger
cliente.update_registered_model(
    MODELO,
    description=("Modelo que predice si una persona desaparecida será localizada "
                 "(Bogotá, Cali, Medellín, Pereira, Barranquilla). Alias: @challenger (último "
                 "entrenado) y @champion (en producción)."))
cliente.update_model_version(
    MODELO, VERSION_CHALLENGER,
    description=f"Iteración {ITERACION}. {motivo} Hiperparámetros: {json.dumps(HIPERPARAMETROS[NOMBRE_GANADOR])}")

tags_version = {
    "algoritmo": NOMBRE_GANADOR,
    "iteracion": ITERACION,
    "hiperparametros": json.dumps(HIPERPARAMETROS[NOMBRE_GANADOR]),
    "metrica_seleccion": METRICA_SELECCION,
    "tabla_baseline": TABLA_BASELINE,
    "tabla_baseline_version": version_baseline,
    "tabla_test": TABLA_TEST,
    "tabla_test_version": version_test,
    "run_id": ID_RUN_PADRE,
    "job_run_id": JOB_RUN_ID,
    "fecha_entrenamiento_utc": FECHA_ENTRENAMIENTO,
    **{f"test_{m}": round(v, 6) for m, v in metricas_ganador.items()},
}
for clave, valor in tags_version.items():
    cliente.set_model_version_tag(MODELO, VERSION_CHALLENGER, clave, str(valor))

cliente.set_registered_model_alias(MODELO, ALIAS_CHALLENGER, VERSION_CHALLENGER)
print(f"✅ {MODELO} versión {VERSION_CHALLENGER} → alias @{ALIAS_CHALLENGER}")

# COMMAND ----------

# CELDA 9: HISTORIAL DE RESULTADOS (tabla_resultados_entrenamiento, APPEND)
pdf_historial = df_resultados.copy()
pdf_historial.insert(0, "iteracion", ITERACION)
pdf_historial["seleccionado"] = pdf_historial["modelo"] == NOMBRE_GANADOR
pdf_historial["hiperparametros"] = pdf_historial["modelo"].map(lambda m: json.dumps(HIPERPARAMETROS[m]))
pdf_historial["version_registrada"] = np.where(pdf_historial["seleccionado"], VERSION_CHALLENGER, "")
pdf_historial["run_comparacion"] = ID_RUN_PADRE
# Trazabilidad de la API: distingue corridas con la misma numeración de iteraciones.
pdf_historial["job_run_id"] = JOB_RUN_ID
# Registros efectivos tras transformación, distribuidos entre entrenamiento y prueba.
pdf_historial["n_train"] = len(X_train)
pdf_historial["n_test"] = len(X_test)
pdf_historial["registros_procesados"] = len(X_train) + len(X_test)
pdf_historial["tabla_entrada_version"] = int(version_entrada)
pdf_historial["fecha_entrenamiento_utc"] = FECHA_ENTRENAMIENTO

(spark.createDataFrame(pdf_historial).write
    .format("delta").mode("append").option("mergeSchema", "true")
    .saveAsTable(TABLA_RESULTADOS))
print(f"✅ {len(pdf_historial)} filas agregadas a {TABLA_RESULTADOS}")
display(df_resultados.round(4))

# COMMAND ----------

# CELDA 10: VERIFICACIÓN — CARGAR EL CHALLENGER POR ALIAS
uri_challenger = f"models:/{MODELO}@{ALIAS_CHALLENGER}"
modelo_cargado = mlflow.sklearn.load_model(uri_challenger)
auc_verificacion = roc_auc_score(y_test, modelo_cargado.predict_proba(X_test)[:, 1])
assert abs(auc_verificacion - metricas_ganador["auc"]) < 1e-6, "El modelo cargado no reproduce el AUC."

print(f"✅ {uri_challenger} reproduce el AUC: {auc_verificacion:.4f}")
print(f"   Run: {HOST}/ml/experiments/{EXPERIMENT_ID}/runs/{ID_RUN_PADRE}")
print(f"   Modelo: {HOST}/explore/data/models/{CATALOGO}/{ESQUEMA}/modelo_desaparecidos")

# COMMAND ----------

# CELDA 11: SALIDA PARA EL PIPELINE (task values)
salida = {"version_challenger": VERSION_CHALLENGER, "algoritmo_challenger": NOMBRE_GANADOR,
          "auc_challenger": round(metricas_ganador["auc"], 6),
          "f1_challenger": round(metricas_ganador["f1"], 6),
          "version_tabla_test": int(version_test), "version_tabla_baseline": int(version_baseline)}
try:
    for clave, valor in salida.items():
        dbutils.jobs.taskValues.set(key=clave, value=valor)
except Exception as e:
    print(f"(Ejecución interactiva: no se publican task values) {e}")
print(json.dumps(salida))