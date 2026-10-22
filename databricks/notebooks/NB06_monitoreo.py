# Databricks notebook source
# MAGIC %md
# MAGIC # NB06_monitoreo — Drift del modelo champion
# MAGIC
# MAGIC **Parámetros de entrada**
# MAGIC - `iteracion` (**requerido**, ≥ 2): identifica el lote nuevo (`lote = iteracion`).
# MAGIC - `umbral_psi_moderado` (default 0.10), `umbral_psi_significativo` (default 0.25), `alfa_ks` (default 0.05), `n_bins` (default 10).
# MAGIC - `job_run_id` (opcional): trazabilidad.
# MAGIC
# MAGIC **Entrada**
# MAGIC - Modelo `modelo_desaparecidos@champion`.
# MAGIC - **Referencia:** `tabla_baseline` con `lote < iteracion` (datos ya conocidos).
# MAGIC - **Datos nuevos:** `tabla_transformacion` con `lote = iteracion` (el lote recién llegado; no es la baseline).
# MAGIC
# MAGIC **Función**
# MAGIC - **PSI por variable categórica** (drift de datos).
# MAGIC - **PSI y KS sobre el puntaje del champion** `P(localizada)`. KS (Kolmogorov-Smirnov) solo aplica a variables continuas, por eso se usa sobre el puntaje.
# MAGIC - Informativo: AUC del champion en la referencia y en el lote nuevo (drift de desempeño).
# MAGIC
# MAGIC | PSI | Nivel |
# MAGIC |---|---|
# MAGIC | < 0.10 | sin drift |
# MAGIC | 0.10 – 0.25 | drift moderado |
# MAGIC | > 0.25 | drift significativo |
# MAGIC
# MAGIC **`hay_drift` = PSI del puntaje > 0.25, o alguna variable con PSI > 0.25, o KS con p-value < 0.05.**
# MAGIC
# MAGIC **Salida**
# MAGIC - Fila en `tabla_monitoreo` (historial, *append*), con `job_run_id`, PSI por variable en JSON y criterios de drift para consulta vía API.
# MAGIC - Run `monitoreo_iterNN` en `MLOps-Desaparecidos/NB06_monitoreo`.
# MAGIC - Task values: `hay_drift`, `nivel_drift`, `psi_puntaje`, `ks_pvalue`.

# COMMAND ----------

# MAGIC %pip install -q mlflow xgboost lightgbm

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# CELDA 1: PARÁMETROS DE ENTRADA
# Los valores predeterminados permiten ejecutar manualmente el notebook.
# En Jobs, los valores enviados por la Task sobrescriben esos predeterminados.
import math

dbutils.widgets.text("iteracion", "2", "Iteración (lote nuevo)")
dbutils.widgets.text("umbral_psi_moderado", "0.10", "Umbral PSI moderado")
dbutils.widgets.text("umbral_psi_significativo", "0.25", "Umbral PSI significativo")
dbutils.widgets.text("alfa_ks", "0.05", "Alfa KS")
dbutils.widgets.text("n_bins", "10", "Bins del puntaje")
dbutils.widgets.text("job_run_id", "", "Id de ejecución del job")


def leer(nombre, tipo):
    """Recibe un parámetro obligatorio y rechaza valores inválidos."""
    valor = dbutils.widgets.get(nombre).strip()
    try:
        resultado = tipo(valor)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Parámetro '{nombre}' inválido: {valor!r}. Debe ser {tipo.__name__}."
        ) from exc
    if isinstance(resultado, float) and not math.isfinite(resultado):
        raise ValueError(f"Parámetro '{nombre}' inválido: {valor!r}. Debe ser finito.")
    return resultado


ITERACION = leer("iteracion", int)
UMBRAL_PSI_MODERADO = leer("umbral_psi_moderado", float)
UMBRAL_PSI_SIGNIFICATIVO = leer("umbral_psi_significativo", float)
ALFA_KS = leer("alfa_ks", float)
N_BINS = leer("n_bins", int)
# En la Task del Job Hijo, asignar este parámetro con {{job.run_id}}.
JOB_RUN_ID = dbutils.widgets.get("job_run_id").strip()

if ITERACION < 2:
    raise ValueError("El monitoreo requiere 'iteracion' >= 2 (en iteración 1 no hay lote anterior).")
if not 0 < UMBRAL_PSI_MODERADO < UMBRAL_PSI_SIGNIFICATIVO:
    raise ValueError("Los umbrales deben cumplir: 0 < PSI moderado < PSI significativo.")
if not 0 < ALFA_KS < 1:
    raise ValueError("El parámetro 'alfa_ks' debe estar entre 0 y 1 (sin incluir extremos).")
if N_BINS < 2:
    raise ValueError("El parámetro 'n_bins' debe ser un entero >= 2.")


# COMMAND ----------

# CELDA 2: IMPORTS Y CONFIGURACIÓN
import json
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
import mlflow
import mlflow.sklearn
from mlflow.models import get_model_info
from mlflow.tracking import MlflowClient
from databricks.sdk import WorkspaceClient
from pyspark.sql import functions as F

warnings.filterwarnings("ignore")

# ⚠️ Si su workspace usa otro catálogo (p. ej. "main"), cámbielo aquí
#    y en los demás notebooks del pipeline.
CATALOGO = "workspace"
ESQUEMA = "mlops_desaparecidos"
MODELO = f"{CATALOGO}.{ESQUEMA}.modelo_desaparecidos"
TABLA_BASELINE = f"{CATALOGO}.{ESQUEMA}.tabla_baseline"
TABLA_NUEVOS = f"{CATALOGO}.{ESQUEMA}.tabla_transformacion"
TABLA_MONITOREO = f"{CATALOGO}.{ESQUEMA}.tabla_monitoreo"
TARGET = "localizada"
EPSILON = 1e-4      # evita log(0) cuando una categoría no aparece en un conjunto

mlflow.set_registry_uri("databricks-uc")
cliente = MlflowClient()

# COMMAND ----------

# CELDA 3: FUNCIONES DE DRIFT (PSI Y NIVEL)
def psi_categorica(referencia, actual, eps=EPSILON):
    """PSI de una variable categórica. Devuelve (psi, detalle por categoría)."""
    categorias = sorted(set(referencia.unique()) | set(actual.unique()))
    p = referencia.value_counts(normalize=True).reindex(categorias, fill_value=0).clip(lower=eps)
    q = actual.value_counts(normalize=True).reindex(categorias, fill_value=0).clip(lower=eps)
    aporte = (q - p) * np.log(q / p)
    detalle = pd.DataFrame({"categoria": categorias, "pct_referencia": p.values,
                            "pct_actual": q.values, "aporte_psi": aporte.values})
    return float(aporte.sum()), detalle


def psi_continua(referencia, actual, n_bins=10, eps=EPSILON):
    """PSI de una variable continua con cortes por cuantiles de la referencia."""
    referencia, actual = np.asarray(referencia, float), np.asarray(actual, float)
    cortes = np.unique(np.quantile(referencia, np.linspace(0, 1, n_bins + 1)))
    if len(cortes) < 3:      # referencia casi constante → cortes uniformes
        minimo = min(referencia.min(), actual.min())
        maximo = max(referencia.max(), actual.max())
        cortes = np.linspace(minimo, maximo + 1e-9, n_bins + 1)
    cortes[0], cortes[-1] = -np.inf, np.inf
    p = np.clip(np.histogram(referencia, cortes)[0] / len(referencia), eps, None)
    q = np.clip(np.histogram(actual, cortes)[0] / len(actual), eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


def nivel_psi(psi, moderado, significativo):
    if psi > significativo:
        return "significativo"
    if psi >= moderado:
        return "moderado"
    return "sin drift"

# COMMAND ----------

# CELDA 4: MODELO CHAMPION
try:
    champion = cliente.get_model_version_by_alias(MODELO, "champion")
except Exception:
    raise AssertionError(f"No existe {MODELO}@champion. Ejecute primero NB05_promocion.")

URI_CHAMPION = f"models:/{MODELO}/{champion.version}"
VARIABLES = get_model_info(URI_CHAMPION).signature.inputs.input_names()
modelo = mlflow.sklearn.load_model(URI_CHAMPION)
print(f"Champion: versión {champion.version} ({champion.tags.get('algoritmo', '?')}) | variables: {VARIABLES}")

# COMMAND ----------

# CELDA 5: DATOS DE REFERENCIA Y DATOS NUEVOS
for tabla in [TABLA_BASELINE, TABLA_NUEVOS]:
    assert spark.catalog.tableExists(tabla), f"No existe {tabla}."

version_baseline = spark.sql(f"DESCRIBE HISTORY {TABLA_BASELINE} LIMIT 1").first()["version"]
version_nuevos = spark.sql(f"DESCRIBE HISTORY {TABLA_NUEVOS} LIMIT 1").first()["version"]

columnas = VARIABLES + [TARGET, "lote"]
pdf_ref = (spark.read.option("versionAsOf", version_baseline).table(TABLA_BASELINE)
             .filter(F.col("lote") < ITERACION).select(*columnas).toPandas())
pdf_new = (spark.read.option("versionAsOf", version_nuevos).table(TABLA_NUEVOS)
             .filter(F.col("lote") == ITERACION).select(*columnas).toPandas())

print(f"Referencia: {len(pdf_ref)} registros (lotes 1–{ITERACION - 1}, {TABLA_BASELINE} v{version_baseline})")
print(f"Nuevos:     {len(pdf_new)} registros (lote {ITERACION}, {TABLA_NUEVOS} v{version_nuevos})")
assert len(pdf_ref) > 0, "No hay datos de referencia (lotes anteriores en tabla_baseline)."
SIN_DATOS_NUEVOS = len(pdf_new) == 0
if SIN_DATOS_NUEVOS:
    print(f"⚠️ El lote {ITERACION} no tiene registros tras la transformación (p. ej. fuera del rango de años). "
          "Se registra el monitoreo sin evaluación de drift.")

# COMMAND ----------

# CELDA 6: CÁLCULO DE DRIFT
def preparar(pdf):
    return pdf[VARIABLES].fillna("Sin información").astype(str)


tabla_psi, detalles_psi = pd.DataFrame(columns=["variable", "psi", "nivel"]), {}
PSI_PUNTAJE = KS_ESTADISTICO = np.nan
KS_PVALUE = 1.0
AUC_REFERENCIA = AUC_NUEVOS = np.nan
puntaje_ref = modelo.predict_proba(preparar(pdf_ref))[:, 1]
puntaje_new = np.array([])

if not SIN_DATOS_NUEVOS:
    X_ref, X_new = preparar(pdf_ref), preparar(pdf_new)

    # ── Drift de datos: PSI por variable ──
    filas = []
    for variable in VARIABLES:
        psi, detalle = psi_categorica(X_ref[variable], X_new[variable])
        detalles_psi[variable] = detalle
        filas.append({"variable": variable, "psi": psi,
                      "nivel": nivel_psi(psi, UMBRAL_PSI_MODERADO, UMBRAL_PSI_SIGNIFICATIVO)})
    tabla_psi = pd.DataFrame(filas).sort_values("psi", ascending=False).reset_index(drop=True)

    # ── Drift del puntaje del modelo: PSI y KS ──
    puntaje_new = modelo.predict_proba(X_new)[:, 1]
    PSI_PUNTAJE = psi_continua(puntaje_ref, puntaje_new, N_BINS)
    resultado_ks = ks_2samp(puntaje_ref, puntaje_new)
    KS_ESTADISTICO, KS_PVALUE = float(resultado_ks.statistic), float(resultado_ks.pvalue)

    # ── Informativo: desempeño en referencia vs. lote nuevo ──
    if pdf_ref[TARGET].nunique() == 2:
        AUC_REFERENCIA = float(roc_auc_score(pdf_ref[TARGET], puntaje_ref))
    if pdf_new[TARGET].nunique() == 2:
        AUC_NUEVOS = float(roc_auc_score(pdf_new[TARGET], puntaje_new))

display(tabla_psi.round(4))

# COMMAND ----------

# CELDA 7: DECISIÓN
PSI_MAX_VARIABLE = float(tabla_psi["psi"].max()) if len(tabla_psi) else 0.0
VARIABLE_PSI_MAX = tabla_psi["variable"].iloc[0] if len(tabla_psi) else ""
variables_drift = tabla_psi.loc[tabla_psi["psi"] > UMBRAL_PSI_SIGNIFICATIVO, "variable"].tolist()

criterios = {
    "psi_puntaje_significativo": bool(not SIN_DATOS_NUEVOS and PSI_PUNTAJE > UMBRAL_PSI_SIGNIFICATIVO),
    "psi_variable_significativo": bool(len(variables_drift) > 0),
    "ks_significativo": bool(not SIN_DATOS_NUEVOS and KS_PVALUE < ALFA_KS),
}
HAY_DRIFT = any(criterios.values())

if SIN_DATOS_NUEVOS:
    NIVEL_DRIFT = "sin datos nuevos"
else:
    peor_psi = max(PSI_PUNTAJE, PSI_MAX_VARIABLE)
    NIVEL_DRIFT = nivel_psi(peor_psi, UMBRAL_PSI_MODERADO, UMBRAL_PSI_SIGNIFICATIVO)
    if HAY_DRIFT and NIVEL_DRIFT != "significativo":
        NIVEL_DRIFT = "significativo (KS)"

print("═" * 60)
print(f"MONITOREO — iteración {ITERACION} — champion v{champion.version}")
print("═" * 60)
print(f"PSI del puntaje:        {PSI_PUNTAJE:.4f}")
print(f"KS del puntaje:         estadístico {KS_ESTADISTICO:.4f} | p-value {KS_PVALUE:.4g}")
print(f"PSI máximo (variable):  {PSI_MAX_VARIABLE:.4f} ({VARIABLE_PSI_MAX})")
print(f"Variables con drift:    {variables_drift or 'ninguna'}")
print(f"AUC referencia / nuevo: {AUC_REFERENCIA:.4f} / {AUC_NUEVOS:.4f}")
print(f"Criterios:              {criterios}")
print(f"➡️ HAY DRIFT: {HAY_DRIFT} ({NIVEL_DRIFT})")

# COMMAND ----------

# CELDA 8: HISTORIAL (tabla_monitoreo, APPEND)
fila = pd.DataFrame([{
    "iteracion": ITERACION,
    "job_run_id": JOB_RUN_ID,
    # PSI individual de todas las variables, serializado para reconstruir psi_variables en la API.
    "psi_variables_json": json.dumps(
        {str(registro["variable"]): float(registro["psi"])
         for _, registro in tabla_psi.iterrows()}, ensure_ascii=False
    ),
    "criterios_json": json.dumps(criterios, ensure_ascii=False),
    "evaluacion_realizada": bool(not SIN_DATOS_NUEVOS),
    "fecha_utc": datetime.now(timezone.utc).isoformat(),
    "version_champion": str(champion.version),
    "algoritmo_champion": champion.tags.get("algoritmo", ""),
    "registros_referencia": len(pdf_ref),
    "registros_nuevos": len(pdf_new),
    "psi_puntaje": float(PSI_PUNTAJE),
    "ks_estadistico": float(KS_ESTADISTICO),
    "ks_pvalue": float(KS_PVALUE),
    "psi_max_variable": PSI_MAX_VARIABLE,
    "variable_psi_max": VARIABLE_PSI_MAX,
    "variables_con_drift": ", ".join(variables_drift),
    "auc_referencia": float(AUC_REFERENCIA),
    "auc_nuevos": float(AUC_NUEVOS),
    "umbral_psi_significativo": UMBRAL_PSI_SIGNIFICATIVO,
    "alfa_ks": ALFA_KS,
    "hay_drift": HAY_DRIFT,
    "nivel_drift": NIVEL_DRIFT,
    "tabla_baseline_version": int(version_baseline),
    "tabla_transformacion_version": int(version_nuevos),
}])

(spark.createDataFrame(fila).write
    .format("delta").mode("append").option("mergeSchema", "true")
    .saveAsTable(TABLA_MONITOREO))
display(spark.table(TABLA_MONITOREO).orderBy("iteracion"))

# COMMAND ----------

# CELDA 9: REGISTRO EN MLFLOW
def contexto_notebook(nombre_por_defecto):
    ruta, host = None, ""
    try:
        ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
        ruta = ctx.notebookPath().get()
        host = "https://" + ctx.browserHostName().get()
    except Exception:
        pass
    return ruta, (ruta.split("/")[-1] if ruta else nombre_por_defecto), host


def finito(valor, defecto=-1.0):
    """MLflow no acepta NaN de forma fiable: se reemplaza por un valor centinela."""
    return float(valor) if np.isfinite(valor) else defecto


RUTA_NOTEBOOK, NOMBRE_NOTEBOOK, HOST = contexto_notebook("NB06_monitoreo")
USUARIO = spark.sql("SELECT current_user()").first()[0]
CARPETA_EXPERIMENTOS = f"/Users/{USUARIO}/MLOps-Desaparecidos"
WorkspaceClient().workspace.mkdirs(CARPETA_EXPERIMENTOS)
EXPERIMENT_ID = mlflow.set_experiment(f"{CARPETA_EXPERIMENTOS}/{NOMBRE_NOTEBOOK}").experiment_id
mlflow.set_experiment_tag("mlflow.experimentKind", "custom_model_development")
mlflow.end_run()

with mlflow.start_run(run_name=f"monitoreo_iter{ITERACION:02d}", experiment_id=EXPERIMENT_ID,
                      description=f"Drift del champion v{champion.version}: lote {ITERACION} vs lotes "
                                  f"1–{ITERACION - 1}. Resultado: {'DRIFT' if HAY_DRIFT else 'sin drift'} "
                                  f"({NIVEL_DRIFT})") as run:
    mlflow.set_tags({
        "etapa_pipeline": "06-monitoreo",
        "iteracion": str(ITERACION),
        "job_run_id": JOB_RUN_ID,
        "notebook": RUTA_NOTEBOOK or NOMBRE_NOTEBOOK,
        "version_champion": str(champion.version),
        "hay_drift": str(HAY_DRIFT),
        "nivel_drift": NIVEL_DRIFT,
    })
    mlflow.log_params({
        "iteracion": ITERACION,
        "version_champion": champion.version,
        "referencia": f"{TABLA_BASELINE} v{version_baseline}, lote < {ITERACION}",
        "datos_nuevos": f"{TABLA_NUEVOS} v{version_nuevos}, lote = {ITERACION}",
        "umbral_psi_moderado": UMBRAL_PSI_MODERADO,
        "umbral_psi_significativo": UMBRAL_PSI_SIGNIFICATIVO,
        "alfa_ks": ALFA_KS,
        "n_bins": N_BINS,
    })
    mlflow.log_metrics({
        "registros_referencia": len(pdf_ref),
        "registros_nuevos": len(pdf_new),
        "psi_puntaje": finito(PSI_PUNTAJE),
        "ks_estadistico": finito(KS_ESTADISTICO),
        "ks_pvalue": finito(KS_PVALUE),
        "psi_max_variable": PSI_MAX_VARIABLE,
        "n_variables_con_drift": len(variables_drift),
        "auc_referencia": finito(AUC_REFERENCIA),
        "auc_nuevos": finito(AUC_NUEVOS),
        "hay_drift": int(HAY_DRIFT),
    })
    for _, f in tabla_psi.iterrows():
        mlflow.log_metric(f"psi_{f['variable']}", float(f["psi"]))
    mlflow.log_dict(criterios, "decision/criterios.json")
    mlflow.log_text(tabla_psi.to_csv(index=False), "drift/psi_por_variable.csv")
    for variable, detalle in detalles_psi.items():
        mlflow.log_text(detalle.to_csv(index=False), f"drift/detalle/{variable}.csv")

    if not SIN_DATOS_NUEVOS:
        fig_puntaje, ax = plt.subplots(figsize=(8, 4))
        bins = np.linspace(0, 1, 21)
        ax.hist(puntaje_ref, bins=bins, alpha=0.55, density=True, label=f"referencia (lotes 1–{ITERACION - 1})")
        ax.hist(puntaje_new, bins=bins, alpha=0.55, density=True, label=f"nuevos (lote {ITERACION})")
        ax.set_xlabel("P(localizada) según el champion")
        ax.set_ylabel("Densidad")
        ax.set_title(f"Puntaje del champion — PSI {PSI_PUNTAJE:.3f} | KS p={KS_PVALUE:.3g}", fontweight="bold")
        ax.legend()
        fig_puntaje.tight_layout()
        mlflow.log_figure(fig_puntaje, "graficas/distribucion_puntaje.png")

        fig_psi, ax = plt.subplots(figsize=(8, 5))
        ax.barh(tabla_psi["variable"], tabla_psi["psi"], color="#2a6f97")
        ax.axvline(UMBRAL_PSI_MODERADO, color="#e9a23b", linestyle="--", label="moderado")
        ax.axvline(UMBRAL_PSI_SIGNIFICATIVO, color="#c0392b", linestyle="--", label="significativo")
        ax.invert_yaxis()
        ax.set_xlabel("PSI")
        ax.set_title(f"PSI por variable — lote {ITERACION}", fontweight="bold")
        ax.legend()
        fig_psi.tight_layout()
        mlflow.log_figure(fig_psi, "graficas/psi_por_variable.png")
        plt.show()

    try:
        mlflow.log_input(mlflow.data.load_delta(table_name=TABLA_BASELINE, version=version_baseline,
                                                name="tabla_baseline"), context="referencia")
        mlflow.log_input(mlflow.data.load_delta(table_name=TABLA_NUEVOS, version=version_nuevos,
                                                name="tabla_transformacion"), context="datos_nuevos")
    except Exception as e:
        print(f"⚠️ No se pudo registrar el dataset Delta: {e}")

print(f"✅ Run {run.info.run_name}: {HOST}/ml/experiments/{EXPERIMENT_ID}/runs/{run.info.run_id}")

# COMMAND ----------

# CELDA 10: SALIDA PARA EL PIPELINE (task values)
salida = {"hay_drift": HAY_DRIFT, "nivel_drift": NIVEL_DRIFT,
          "psi_puntaje": finito(PSI_PUNTAJE), "ks_pvalue": finito(KS_PVALUE),
          "version_champion": str(champion.version)}
try:
    for clave, valor in salida.items():
        dbutils.jobs.taskValues.set(key=clave, value=valor)
except Exception as e:
    print(f"(Ejecución interactiva: no se publican task values) {e}")
print(json.dumps(salida, ensure_ascii=False))