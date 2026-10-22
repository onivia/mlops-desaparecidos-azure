# Databricks notebook source
# MAGIC %md
# MAGIC # NB05_promocion — Challenger vs. champion
# MAGIC
# MAGIC **Parámetros de entrada**
# MAGIC - `tolerancia_auc` (default 0.0001): diferencia de AUC por debajo de la cual se considera **empate**.
# MAGIC - `iteracion` (default 1) y `job_run_id` (opcional): para nombrar el run y la trazabilidad.
# MAGIC
# MAGIC **Entrada:** `modelo_desaparecidos@challenger`, `modelo_desaparecidos@champion` (si existe) y la tabla `tabla_test`.
# MAGIC
# MAGIC **Función:** se **reevalúan los dos modelos sobre la misma `tabla_test`** y se aplica la regla:
# MAGIC
# MAGIC | Caso | Resultado |
# MAGIC |---|---|
# MAGIC | No existe `@champion` (primera iteración) | se promueve el challenger |
# MAGIC | AUC challenger − AUC champion > tolerancia | se promueve el challenger |
# MAGIC | Empate de AUC (diferencia ≤ tolerancia) y F1 challenger ≥ F1 champion | se promueve el challenger |
# MAGIC | Cualquier otro caso | se mantiene el champion |
# MAGIC
# MAGIC Revisar si existe un champion es parte de la función de este notebook (no es control de flujo): por eso la primera iteración no necesita una rama especial en el pipeline.
# MAGIC
# MAGIC **Salida**
# MAGIC - Alias `@champion` apuntando al ganador.
# MAGIC - Fila en `tabla_promocion` (historial, *append*) con métricas de ambos modelos, diferencias, ganador, `job_run_id` y versión resultante, para consulta vía API.
# MAGIC - Run `promocion_iterNN` en `MLOps-Desaparecidos/NB05_promocion`.
# MAGIC - Task values: `promovido`, `version_champion`, `auc_champion`.

# COMMAND ----------

# MAGIC %pip install -q mlflow xgboost lightgbm

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# CELDA 1: PARÁMETROS DE ENTRADA
# Los valores predeterminados permiten ejecutar manualmente el notebook.
# En Jobs, los parámetros suministrados por la Task sobrescriben estos valores.
import math

dbutils.widgets.text("tolerancia_auc", "0.0001", "Tolerancia de empate en AUC")
dbutils.widgets.text("iteracion", "1", "Iteración")
dbutils.widgets.text("job_run_id", "", "Id de ejecución del job")

valor_tolerancia = dbutils.widgets.get("tolerancia_auc").strip()
try:
    TOLERANCIA_AUC = float(valor_tolerancia)
except ValueError as exc:
    raise ValueError(
        f"Parámetro 'tolerancia_auc' inválido: {valor_tolerancia!r}. Debe ser un número no negativo."
    ) from exc
if not math.isfinite(TOLERANCIA_AUC) or TOLERANCIA_AUC < 0:
    raise ValueError(
        f"Parámetro 'tolerancia_auc' inválido: {valor_tolerancia!r}. Debe ser finito y no negativo."
    )

valor_iteracion = dbutils.widgets.get("iteracion").strip()
try:
    ITERACION = int(valor_iteracion)
except ValueError as exc:
    raise ValueError(
        f"Parámetro 'iteracion' inválido: {valor_iteracion!r}. Debe ser un entero positivo."
    ) from exc
if ITERACION <= 0:
    raise ValueError(
        f"Parámetro 'iteracion' inválido: {ITERACION}. Debe ser mayor que cero."
    )

# En el Job Hijo, configurar 'job_run_id' mediante {{job.run_id}}.
JOB_RUN_ID = dbutils.widgets.get("job_run_id").strip()


# COMMAND ----------

# CELDA 2: IMPORTS Y CONFIGURACIÓN
import json
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import get_model_info
from mlflow.tracking import MlflowClient
from databricks.sdk import WorkspaceClient
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score, precision_score,
                             recall_score, average_precision_score)

warnings.filterwarnings("ignore")

# ⚠️ Si su workspace usa otro catálogo (p. ej. "main"), cámbielo aquí
#    y en los demás notebooks del pipeline.
CATALOGO = "workspace"
ESQUEMA = "mlops_desaparecidos"
MODELO = f"{CATALOGO}.{ESQUEMA}.modelo_desaparecidos"
TABLA_TEST = f"{CATALOGO}.{ESQUEMA}.tabla_test"
TABLA_PROMOCION = f"{CATALOGO}.{ESQUEMA}.tabla_promocion"
TARGET = "localizada"

mlflow.set_registry_uri("databricks-uc")
cliente = MlflowClient()

# COMMAND ----------

# CELDA 3: REGLA DE PROMOCIÓN (función pura, sin efectos)
def decidir_promocion(metricas_challenger, metricas_champion, tolerancia):
    """Devuelve (promover, motivo). metricas_champion = None si no hay champion."""
    if metricas_champion is None:
        return True, "No existía champion: el challenger se convierte en champion."
    dif_auc = metricas_challenger["auc"] - metricas_champion["auc"]
    if dif_auc > tolerancia:
        return True, f"AUC del challenger supera al champion ({dif_auc:+.4f})."
    if abs(dif_auc) <= tolerancia:
        dif_f1 = metricas_challenger["f1"] - metricas_champion["f1"]
        if dif_f1 >= 0:
            return True, f"Empate en AUC ({dif_auc:+.4f}); F1 del challenger igual o mejor ({dif_f1:+.4f})."
        return False, f"Empate en AUC ({dif_auc:+.4f}); F1 del champion es mejor ({dif_f1:+.4f})."
    return False, f"AUC del champion es mejor ({dif_auc:+.4f}): se mantiene."

# COMMAND ----------

# CELDA 4: MODELOS DE ENTRADA
def version_por_alias(alias):
    try:
        return cliente.get_model_version_by_alias(MODELO, alias)
    except Exception:
        return None


challenger = version_por_alias("challenger")
assert challenger is not None, (
    f"No existe {MODELO}@challenger. Ejecute primero NB04_entrenamiento.")
champion = version_por_alias("champion")

print(f"Challenger: versión {challenger.version} ({challenger.tags.get('algoritmo', '?')})")
print("Champion:   " + (f"versión {champion.version} ({champion.tags.get('algoritmo', '?')})"
                         if champion else "no existe (primera iteración o reinicio)"))

# COMMAND ----------

# CELDA 5: EVALUACIÓN SOBRE LA MISMA tabla_test
assert spark.catalog.tableExists(TABLA_TEST), f"No existe {TABLA_TEST}. Ejecute primero NB04_entrenamiento."
version_test = spark.sql(f"DESCRIBE HISTORY {TABLA_TEST} LIMIT 1").first()["version"]
pdf_test = spark.read.option("versionAsOf", version_test).table(TABLA_TEST).toPandas()
y_test = pdf_test[TARGET].astype(int)
assert y_test.nunique() == 2, "tabla_test tiene una sola clase: no se puede calcular el AUC."


def evaluar(version_modelo):
    """Carga una versión del modelo y la evalúa con sus propias variables de entrada."""
    uri = f"models:/{MODELO}/{version_modelo.version}"
    variables = get_model_info(uri).signature.inputs.input_names()
    faltantes = [v for v in variables if v not in pdf_test.columns]
    assert not faltantes, f"tabla_test no tiene las variables del modelo v{version_modelo.version}: {faltantes}"
    X = pdf_test[variables].fillna("Sin información").astype(str)
    modelo = mlflow.sklearn.load_model(uri)
    y_prob = modelo.predict_proba(X)[:, 1]
    y_pred = modelo.predict(X)
    return {
        "auc": float(roc_auc_score(y_test, y_prob)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "auc_pr": float(average_precision_score(y_test, y_prob)),
    }


metricas_challenger = evaluar(challenger)
mismo_modelo = champion is not None and champion.version == challenger.version
metricas_champion = (metricas_challenger if mismo_modelo
                     else evaluar(champion) if champion is not None else None)

comparacion = pd.DataFrame({"challenger": metricas_challenger,
                            "champion": metricas_champion or {k: np.nan for k in metricas_challenger}})
comparacion["diferencia"] = comparacion["challenger"] - comparacion["champion"]
print(f"Evaluación sobre {TABLA_TEST} v{version_test} ({len(pdf_test)} registros)")
display(comparacion.round(4).reset_index(names="metrica"))

# COMMAND ----------

# CELDA 6: DECISIÓN Y ACTUALIZACIÓN DEL ALIAS @champion
if mismo_modelo:
    PROMOVIDO, MOTIVO = False, "El challenger ya es el champion (misma versión): no hay cambio."
else:
    PROMOVIDO, MOTIVO = decidir_promocion(metricas_challenger, metricas_champion, TOLERANCIA_AUC)

version_champion_anterior = champion.version if champion else None
if PROMOVIDO:
    cliente.set_registered_model_alias(MODELO, "champion", challenger.version)
    cliente.set_model_version_tag(MODELO, challenger.version, "champion_desde_iteracion", str(ITERACION))
    VERSION_CHAMPION = str(challenger.version)
else:
    VERSION_CHAMPION = str(champion.version)

metricas_champion_final = metricas_challenger if PROMOVIDO else metricas_champion
print(("✅ PROMOVIDO" if PROMOVIDO else "⏸️ SE MANTIENE EL CHAMPION") + f" — {MOTIVO}")
print(f"   {MODELO}@champion → versión {VERSION_CHAMPION}")

# COMMAND ----------

# CELDA 7: HISTORIAL (tabla_promocion, APPEND)
fila = pd.DataFrame([{
    "iteracion": ITERACION,
    "fecha_utc": datetime.now(timezone.utc).isoformat(),
    "version_challenger": str(challenger.version),
    "algoritmo_challenger": challenger.tags.get("algoritmo", ""),
    "version_champion_anterior": str(version_champion_anterior or ""),
    "algoritmo_champion_anterior": champion.tags.get("algoritmo", "") if champion else "",
    "auc_challenger": metricas_challenger["auc"],
    "f1_challenger": metricas_challenger["f1"],
    "auc_champion_anterior": metricas_champion["auc"] if metricas_champion else None,
    "f1_champion_anterior": metricas_champion["f1"] if metricas_champion else None,
    "tolerancia_auc": TOLERANCIA_AUC,
    "promovido": PROMOVIDO,
    "motivo": MOTIVO,
    "version_champion_resultante": VERSION_CHAMPION,
    "tabla_test_version": int(version_test),
}])
fila = fila.astype({"auc_champion_anterior": "float64", "f1_champion_anterior": "float64"})
# Campos adicionales para consolidar la respuesta JSON de la API.
# Ambas alternativas ya fueron evaluadas sobre la MISMA tabla_test.
fila["job_run_id"] = JOB_RUN_ID
fila["ganador"] = "challenger" if PROMOVIDO else "champion"
fila["algoritmo_champion_resultante"] = (
    challenger.tags.get("algoritmo", "") if PROMOVIDO else champion.tags.get("algoritmo", "")
)
fila["criterio_principal"] = "auc"
fila["criterio_desempate"] = "f1"
for metrica in ("accuracy", "precision", "recall", "auc_pr"):
    fila[f"{metrica}_challenger"] = float(metricas_challenger[metrica])
    fila[f"{metrica}_champion_anterior"] = (
        float(metricas_champion[metrica]) if metricas_champion is not None else np.nan
    )
fila["diferencia_auc"] = (
    float(metricas_challenger["auc"] - metricas_champion["auc"])
    if metricas_champion is not None else np.nan
)
fila["diferencia_f1"] = (
    float(metricas_challenger["f1"] - metricas_champion["f1"])
    if metricas_champion is not None else np.nan
)

(spark.createDataFrame(fila).write
    .format("delta").mode("append").option("mergeSchema", "true")
    .saveAsTable(TABLA_PROMOCION))
display(spark.table(TABLA_PROMOCION).orderBy("iteracion"))

# COMMAND ----------

# CELDA 8: REGISTRO EN MLFLOW
def contexto_notebook(nombre_por_defecto):
    ruta, host = None, ""
    try:
        ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
        ruta = ctx.notebookPath().get()
        host = "https://" + ctx.browserHostName().get()
    except Exception:
        pass
    return ruta, (ruta.split("/")[-1] if ruta else nombre_por_defecto), host


RUTA_NOTEBOOK, NOMBRE_NOTEBOOK, HOST = contexto_notebook("NB05_promocion")
USUARIO = spark.sql("SELECT current_user()").first()[0]
CARPETA_EXPERIMENTOS = f"/Users/{USUARIO}/MLOps-Desaparecidos"
WorkspaceClient().workspace.mkdirs(CARPETA_EXPERIMENTOS)
EXPERIMENT_ID = mlflow.set_experiment(f"{CARPETA_EXPERIMENTOS}/{NOMBRE_NOTEBOOK}").experiment_id
mlflow.set_experiment_tag("mlflow.experimentKind", "custom_model_development")
mlflow.end_run()

with mlflow.start_run(run_name=f"promocion_iter{ITERACION:02d}", experiment_id=EXPERIMENT_ID,
                      description=f"Challenger v{challenger.version} vs champion "
                                  f"v{version_champion_anterior or '—'}: {MOTIVO}") as run:
    mlflow.set_tags({
        "etapa_pipeline": "05-promocion",
        "iteracion": str(ITERACION),
        "job_run_id": JOB_RUN_ID,
        "notebook": RUTA_NOTEBOOK or NOMBRE_NOTEBOOK,
        "decision": "promovido" if PROMOVIDO else "mantenido",
        "motivo": MOTIVO,
        "version_champion": VERSION_CHAMPION,
    })
    mlflow.log_params({
        "iteracion": ITERACION,
        "modelo": MODELO,
        "version_challenger": challenger.version,
        "version_champion_anterior": version_champion_anterior or "ninguno",
        "metrica_principal": "auc",
        "metrica_desempate": "f1",
        "tolerancia_auc": TOLERANCIA_AUC,
        "tabla_test_version": version_test,
    })
    mlflow.log_metrics({f"challenger_{k}": v for k, v in metricas_challenger.items()})
    if metricas_champion is not None:
        mlflow.log_metrics({f"champion_anterior_{k}": v for k, v in metricas_champion.items()})
    mlflow.log_metrics({f"champion_{k}": v for k, v in metricas_champion_final.items()})
    mlflow.log_metric("promovido", int(PROMOVIDO))
    mlflow.log_text(comparacion.reset_index(names="metrica").to_csv(index=False), "comparacion.csv")

    fig, ax = plt.subplots(figsize=(9, 4))
    nombres = list(metricas_challenger)
    x = np.arange(len(nombres))
    ax.bar(x - 0.2, [metricas_challenger[m] for m in nombres], 0.4, label=f"challenger v{challenger.version}")
    if metricas_champion is not None and not mismo_modelo:
        ax.bar(x + 0.2, [metricas_champion[m] for m in nombres], 0.4,
               label=f"champion v{version_champion_anterior}")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in nombres])
    ax.set_ylim(0, 1)
    ax.set_title(f"Challenger vs champion — iteración {ITERACION}", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    mlflow.log_figure(fig, "graficas/challenger_vs_champion.png")
    plt.show()
    try:
        mlflow.log_input(mlflow.data.load_delta(table_name=TABLA_TEST, version=version_test,
                                                name="tabla_test"), context="evaluacion")
    except Exception as e:
        print(f"⚠️ No se pudo registrar el dataset Delta: {e}")

print(f"✅ Run {run.info.run_name}: {HOST}/ml/experiments/{EXPERIMENT_ID}/runs/{run.info.run_id}")

# COMMAND ----------

# CELDA 9: SALIDA PARA EL PIPELINE (task values)
salida = {"promovido": PROMOVIDO, "version_champion": VERSION_CHAMPION,
          "auc_champion": round(metricas_champion_final["auc"], 6), "motivo": MOTIVO}
try:
    for clave, valor in salida.items():
        dbutils.jobs.taskValues.set(key=clave, value=valor)
except Exception as e:
    print(f"(Ejecución interactiva: no se publican task values) {e}")
print(json.dumps(salida, ensure_ascii=False))