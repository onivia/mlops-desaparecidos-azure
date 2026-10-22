# Databricks notebook source
# MAGIC %md
# MAGIC # NB00_configuracion — Configuración del ciclo de iteraciones
# MAGIC
# MAGIC Notebook de **orquestación** (no de negocio). Lo ejecuta el job padre `pipeline_mlops_desaparecidos` antes del bucle *For each*.
# MAGIC
# MAGIC **Parámetros de entrada**
# MAGIC - `max_iteraciones` (default 5): número de iteraciones del ciclo.
# MAGIC - `tamano_lote` (default 1000): registros nuevos por iteración. La iteración *n* extrae `n × tamano_lote` registros.
# MAGIC - `reiniciar_modelos` (default `false`): si es `true`, borra los alias `@champion` / `@challenger` y las tablas de historial, para que la iteración 1 empiece **sin modelos**.
# MAGIC
# MAGIC **Salida**
# MAGIC - Task value `iteraciones`: lista `[{"iteracion": 1, "limite_registros": 1000, "tamano_lote": 1000}, …]` que consume el *For each*.
# MAGIC - Run `configuracion_pipeline` en el experimento `MLOps-Desaparecidos/NB00_configuracion`.

# COMMAND ----------

# MAGIC %pip install -q mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# CELDA 1: PARÁMETROS DE ENTRADA
dbutils.widgets.text("max_iteraciones", "5", "Máximo de iteraciones")
dbutils.widgets.text("tamano_lote", "1000", "Tamaño del lote")
dbutils.widgets.dropdown("reiniciar_modelos", "false", ["true", "false"], "Reiniciar modelos")


def entero_positivo(nombre, defecto):
    """Lee un widget y lo convierte a entero > 0 (usa el default si no es válido)."""
    valor = dbutils.widgets.get(nombre).strip()
    try:
        numero = int(valor)
        if numero <= 0:
            raise ValueError
        return numero
    except ValueError:
        print(f"⚠️ '{nombre}' no válido ('{valor}'): se usa {defecto}.")
        return defecto


MAX_ITERACIONES = entero_positivo("max_iteraciones", 5)
TAMANO_LOTE = entero_positivo("tamano_lote", 1000)
REINICIAR = dbutils.widgets.get("reiniciar_modelos").strip().lower() == "true"

print(f"Máximo de iteraciones: {MAX_ITERACIONES} | Tamaño de lote: {TAMANO_LOTE} | Reiniciar: {REINICIAR}")

# COMMAND ----------

# CELDA 2: CONFIGURACIÓN COMÚN DEL PIPELINE
# ⚠️ Si su workspace usa otro catálogo (p. ej. "main"), cámbielo aquí
#    y en los demás notebooks del pipeline.
import mlflow
from mlflow.tracking import MlflowClient

CATALOGO = "workspace"
ESQUEMA = "mlops_desaparecidos"
MODELO = f"{CATALOGO}.{ESQUEMA}.modelo_desaparecidos"
ALIASES = ["champion", "challenger"]
TABLAS_HISTORIAL = [
    f"{CATALOGO}.{ESQUEMA}.tabla_promocion",
    f"{CATALOGO}.{ESQUEMA}.tabla_monitoreo",
    f"{CATALOGO}.{ESQUEMA}.tabla_resultados_entrenamiento",
]

spark.sql(f"""
    CREATE SCHEMA IF NOT EXISTS {CATALOGO}.{ESQUEMA}
    COMMENT 'Pipeline MLOps - Desaparecidos en Bogotá, Cali, Medellín, Pereira y Barranquilla'
""")
mlflow.set_registry_uri("databricks-uc")
cliente = MlflowClient()

# COMMAND ----------

# CELDA 3: REINICIO (OPCIONAL)
# Sin alias, la iteración 1 arranca "sin modelos": la promoción
# convertirá al primer challenger en champion.
# Las versiones del modelo NO se borran: quedan para trazabilidad.
acciones_reinicio = []

if REINICIAR:
    for alias in ALIASES:
        try:
            cliente.delete_registered_model_alias(MODELO, alias)
            acciones_reinicio.append(f"alias @{alias} eliminado")
        except Exception:
            acciones_reinicio.append(f"alias @{alias} no existía")
    for tabla in TABLAS_HISTORIAL:
        spark.sql(f"DROP TABLE IF EXISTS {tabla}")
        acciones_reinicio.append(f"tabla {tabla} eliminada")
else:
    acciones_reinicio.append("sin reinicio: se conservan el champion y el historial")

for accion in acciones_reinicio:
    print(f"• {accion}")

# COMMAND ----------

# CELDA 4: LISTA DE ITERACIONES PARA EL FOR EACH
import json

# tamano_lote viaja en cada iteración: NB01 lo necesita para numerar los lotes
iteraciones = [
    {"iteracion": str(i), "limite_registros": str(i * TAMANO_LOTE), "tamano_lote": str(TAMANO_LOTE)}
    for i in range(1, MAX_ITERACIONES + 1)
]

try:
    dbutils.jobs.taskValues.set(key="iteraciones", value=iteraciones)
    dbutils.jobs.taskValues.set(key="tamano_lote", value=TAMANO_LOTE)
except Exception as e:
    print(f"(Ejecución interactiva: no se publican task values) {e}")

print(json.dumps(iteraciones, indent=2))

# COMMAND ----------

# CELDA 5: REGISTRO EN MLFLOW
from databricks.sdk import WorkspaceClient

try:
    RUTA_NOTEBOOK = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    NOMBRE_NOTEBOOK = RUTA_NOTEBOOK.split("/")[-1]
except Exception:
    RUTA_NOTEBOOK, NOMBRE_NOTEBOOK = None, "NB00_configuracion"

USUARIO = spark.sql("SELECT current_user()").first()[0]
CARPETA_EXPERIMENTOS = f"/Users/{USUARIO}/MLOps-Desaparecidos"
WorkspaceClient().workspace.mkdirs(CARPETA_EXPERIMENTOS)
EXPERIMENT_ID = mlflow.set_experiment(f"{CARPETA_EXPERIMENTOS}/{NOMBRE_NOTEBOOK}").experiment_id
mlflow.end_run()

with mlflow.start_run(run_name="configuracion_pipeline", experiment_id=EXPERIMENT_ID,
                      description="Configuración del ciclo de iteraciones del pipeline MLOps") as run:
    mlflow.set_tags({"etapa_pipeline": "00-configuracion",
                     "notebook": RUTA_NOTEBOOK or NOMBRE_NOTEBOOK})
    mlflow.log_params({"max_iteraciones": MAX_ITERACIONES, "tamano_lote": TAMANO_LOTE,
                       "reiniciar_modelos": REINICIAR, "modelo": MODELO})
    mlflow.log_dict({"iteraciones": iteraciones, "reinicio": acciones_reinicio},
                    "configuracion.json")

print(f"✅ Configuración registrada (run {run.info.run_id})")