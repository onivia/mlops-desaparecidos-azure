# Databricks notebook source
# MAGIC %md
# MAGIC # NB01_extraccion — Extracción de datos de origen
# MAGIC
# MAGIC Consume el API de **datos.gov.co** (dataset `8hqm-7fdt`, Desaparecidos en Colombia) para Bogotá, Cali, Medellín, Pereira y Barranquilla.
# MAGIC
# MAGIC **Parámetros de entrada**
# MAGIC - `limite_registros` (default 1000): registros a solicitar al API. El pipeline envía `iteracion × tamano_lote`.
# MAGIC - `tamano_lote` (default 1000): tamaño de cada lote; se usa para numerar los lotes.
# MAGIC - `iteracion` (default 1): solo para nombrar el run y etiquetar la trazabilidad.
# MAGIC - `job_run_id` (opcional): id de ejecución del job, para enlazar los runs de una iteración. En la Task del Job Hijo se debe pasar como `{{job.run_id}}` para registrar el id real; en una ejecución manual puede quedar vacío.
# MAGIC
# MAGIC **Función**
# MAGIC - Se consulta el API con `$order=:id`: el orden es estable, así que la extracción de 2000 registros contiene los mismos 1000 de la anterior más 1000 nuevos.
# MAGIC - Se agregan dos columnas: `posicion` (1, 2, 3…) y `lote` = ⌈posicion / tamano_lote⌉. Con `lote` el monitoreo distingue los datos nuevos de los ya vistos.
# MAGIC
# MAGIC **Salida**
# MAGIC - Tabla Delta `workspace.mlops_desaparecidos.tabla_raw` (sobrescrita en cada ejecución).
# MAGIC - Run `extraccion_iterNN` en el experimento `MLOps-Desaparecidos/NB01_extraccion`.
# MAGIC - Task values: `registros_extraidos`, `lote_max`, `version_tabla_raw`.
# MAGIC
# MAGIC **Validaciones de seguridad**
# MAGIC - Los parámetros numéricos deben ser enteros positivos; un valor inválido detiene la ejecución.
# MAGIC - La API debe devolver exactamente `limite_registros` registros; si devuelve menos, la ejecución termina antes de sobrescribir `tabla_raw`.
# MAGIC

# COMMAND ----------

# MAGIC %pip install -q mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# CELDA 1: PARÁMETROS DE ENTRADA
dbutils.widgets.text("limite_registros", "1000", "Límite de registros")
dbutils.widgets.text("tamano_lote", "1000", "Tamaño del lote")
dbutils.widgets.text("iteracion", "1", "Iteración")
# El Job Hijo deberá enviar job_run_id con la referencia dinámica {{job.run_id}}.
# En una ejecución manual puede quedar vacío.
dbutils.widgets.text("job_run_id", "", "Id de ejecución del job")


def entero_positivo(nombre):
    """Lee un widget y exige un entero positivo; falla si es inválido."""
    valor = dbutils.widgets.get(nombre).strip()
    try:
        numero = int(valor)
    except ValueError as e:
        raise ValueError(f"Parámetro '{nombre}' inválido: '{valor}'. Debe ser un entero positivo.") from e
    if numero <= 0:
        raise ValueError(f"Parámetro '{nombre}' inválido: '{valor}'. Debe ser mayor que cero.")
    return numero


LIMITE_REGISTROS = entero_positivo("limite_registros")
TAMANO_LOTE = entero_positivo("tamano_lote")
ITERACION = entero_positivo("iteracion")
JOB_RUN_ID = dbutils.widgets.get("job_run_id").strip()

print(f"Iteración {ITERACION}: se extraen hasta {LIMITE_REGISTROS} registros (lotes de {TAMANO_LOTE})")

# COMMAND ----------

# CELDA 2: IMPORTS Y CONFIGURACIÓN
import time
import json
import requests
import numpy as np
import pandas as pd
import mlflow
from datetime import datetime, timezone
from pyspark.sql.types import StructType, StructField, StringType, LongType, IntegerType

# ⚠️ Si su workspace usa otro catálogo (p. ej. "main"), cámbielo aquí
#    y en los demás notebooks del pipeline.
CATALOGO = "workspace"
ESQUEMA = "mlops_desaparecidos"
TABLA_RAW = f"{CATALOGO}.{ESQUEMA}.tabla_raw"   # Salida de este notebook

BASE_URL = "https://www.datos.gov.co/resource/8hqm-7fdt.json"
CIUDADES = {
    "Bogotá": "11001",
    "Cali": "76001",
    "Medellín": "5001",
    "Pereira": "66001",
    "Barranquilla": "8001",
}

# COMMAND ----------

# CELDA 3: CONSULTA AL API
# $order=:id → orden estable (identificador interno de la fila en
# datos.gov.co): cada extracción es la anterior + los registros nuevos.
codigos = "', '".join(CIUDADES.values())
parametros_api = {
    "$limit": LIMITE_REGISTROS,
    "$order": ":id",
    "$where": f"codigo_dane_municipio in ('{codigos}')",
}

fecha_extraccion = datetime.now(timezone.utc)
data = None
for intento in range(1, 4):                       # hasta 3 intentos
    try:
        respuesta = requests.get(BASE_URL, params=parametros_api, timeout=600)
        respuesta.raise_for_status()
        data = respuesta.json()
        break
    except Exception as e:
        print(f"⚠️ Intento {intento} fallido: {e}")
        if intento == 3:
            raise
        time.sleep(10 * intento)

assert isinstance(data, list), f"Respuesta inesperada del API: {str(data)[:300]}"
assert len(data) > 0, "El API no devolvió registros."
if len(data) != LIMITE_REGISTROS:
    raise RuntimeError(
        f"Extracción incompleta: se solicitaron {LIMITE_REGISTROS} registros "
        f"pero el API devolvió {len(data)}. No se actualizará {TABLA_RAW}."
    )
print(f"Registros recibidos: {len(data)} (solicitados: {LIMITE_REGISTROS})")

# COMMAND ----------

# CELDA 4: DATAFRAME CON POSICIÓN Y LOTE
pdf_extraccion = pd.DataFrame(data)
columnas_api = list(pdf_extraccion.columns)

# Todas las columnas del API como texto: el tipo no depende de la muestra
pdf_extraccion = pdf_extraccion.astype(object).where(pdf_extraccion.notna(), None)

# Posición en el orden del API y lote al que pertenece cada registro
pdf_extraccion["posicion"] = np.arange(1, len(pdf_extraccion) + 1, dtype="int64")
pdf_extraccion["lote"] = ((pdf_extraccion["posicion"] - 1) // TAMANO_LOTE + 1).astype("int32")

esquema = StructType(
    [StructField(c, StringType(), True) for c in columnas_api]
    + [StructField("posicion", LongType(), False), StructField("lote", IntegerType(), False)]
)
df = spark.createDataFrame(pdf_extraccion, schema=esquema)

# ── Validaciones mínimas para las etapas siguientes ──
assert "id" in df.columns, "El API no devolvió la columna 'id' (se usa como id_registro)."
n_ids = df.select("id").distinct().count()
assert n_ids == len(data), f"La columna 'id' no es única: {n_ids} valores para {len(data)} registros."

LOTE_MAX = int(pdf_extraccion["lote"].max())
conteo_ciudades = {c: df.filter(df.codigo_dane_municipio == cod).count() for c, cod in CIUDADES.items()}
conteo_lotes = pdf_extraccion["lote"].value_counts().sort_index()

for ciudad, n in conteo_ciudades.items():
    print(f"{ciudad:<14} {CIUDADES[ciudad]:>6} -> {n} registros")
print(f"Lotes: {conteo_lotes.to_dict()}")

# COMMAND ----------

# CELDA 5: GUARDAR tabla_raw (SOBRESCRITURA)
# Delta conserva las versiones anteriores en su historial (time travel).
spark.sql(f"""
    CREATE SCHEMA IF NOT EXISTS {CATALOGO}.{ESQUEMA}
    COMMENT 'Pipeline MLOps - Desaparecidos en Bogotá, Cali, Medellín, Pereira y Barranquilla'
""")

(df.write
   .format("delta")
   .mode("overwrite")
   .option("overwriteSchema", "true")
   .saveAsTable(TABLA_RAW))

spark.sql(f"""
    COMMENT ON TABLE {TABLA_RAW} IS
    'Extracción del API datos.gov.co 8hqm-7fdt generada por NB01_extraccion (iteración {ITERACION}, límite {LIMITE_REGISTROS})'
""")
spark.sql(f"""
    ALTER TABLE {TABLA_RAW} SET TBLPROPERTIES (
        'extraccion.iteracion' = '{ITERACION}',
        'extraccion.limite_registros' = '{LIMITE_REGISTROS}',
        'extraccion.tamano_lote' = '{TAMANO_LOTE}',
        'extraccion.registros_obtenidos' = '{len(data)}',
        'extraccion.fecha_utc' = '{fecha_extraccion.isoformat()}'
    )
""")

version_raw = spark.sql(f"DESCRIBE HISTORY {TABLA_RAW} LIMIT 1").first()["version"]
n_tabla = spark.table(TABLA_RAW).count()
assert n_tabla == len(data), f"La tabla tiene {n_tabla} registros, se esperaban {len(data)}."

print(f"✅ {TABLA_RAW} | versión {version_raw} | {n_tabla} registros | {len(df.columns)} columnas")
display(spark.table(TABLA_RAW).limit(5))

# COMMAND ----------

# CELDA 6: REGISTRO EN MLFLOW
import re
from databricks.sdk import WorkspaceClient


def contexto_notebook(nombre_por_defecto):
    """Ruta, nombre del notebook y host del workspace (con valores de respaldo)."""
    ruta, host = None, ""
    try:
        ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
        ruta = ctx.notebookPath().get()
        host = "https://" + ctx.browserHostName().get()
    except Exception:
        pass
    return ruta, (ruta.split("/")[-1] if ruta else nombre_por_defecto), host


def nombre_metrica(texto):
    return re.sub(r"[^\w]+", "_", str(texto).lower()).strip("_")


RUTA_NOTEBOOK, NOMBRE_NOTEBOOK, HOST = contexto_notebook("NB01_extraccion")
USUARIO = spark.sql("SELECT current_user()").first()[0]
CARPETA_EXPERIMENTOS = f"/Users/{USUARIO}/MLOps-Desaparecidos"
WorkspaceClient().workspace.mkdirs(CARPETA_EXPERIMENTOS)

mlflow.set_registry_uri("databricks-uc")
EXPERIMENT_ID = mlflow.set_experiment(f"{CARPETA_EXPERIMENTOS}/{NOMBRE_NOTEBOOK}").experiment_id
mlflow.set_experiment_tag("mlflow.experimentKind", "custom_model_development")
mlflow.end_run()

with mlflow.start_run(run_name=f"extraccion_iter{ITERACION:02d}", experiment_id=EXPERIMENT_ID,
                      description=f"Extracción de {LIMITE_REGISTROS} registros del API → `{TABLA_RAW}`") as run:
    mlflow.set_tags({
        "etapa_pipeline": "01-extraccion",
        "iteracion": str(ITERACION),
        "job_run_id": JOB_RUN_ID,
        "notebook": RUTA_NOTEBOOK or NOMBRE_NOTEBOOK,
        "tabla_salida": TABLA_RAW,
        "tabla_salida_version": str(version_raw),
    })
    mlflow.log_params({
        "iteracion": ITERACION,
        "limite_registros": LIMITE_REGISTROS,
        "tamano_lote": TAMANO_LOTE,
        "url_api": BASE_URL,
        "orden_api": ":id",
        "ciudades": ", ".join(CIUDADES),
        "codigos_dane": ", ".join(CIUDADES.values()),
        "fecha_extraccion_utc": fecha_extraccion.isoformat(),
        "tabla_salida": TABLA_RAW,
        "tabla_salida_version": version_raw,
    })
    mlflow.log_metrics({
        "registros_extraidos": len(data),
        "n_columnas": len(df.columns),
        "lote_max": LOTE_MAX,
        "registros_lote_actual": int(conteo_lotes.get(LOTE_MAX, 0)),
        "limite_alcanzado": int(len(data) >= LIMITE_REGISTROS),
    })
    for ciudad, n in conteo_ciudades.items():
        mlflow.log_metric(f"registros_{nombre_metrica(ciudad)}", n)

    mlflow.log_dict({c: t for c, t in df.dtypes}, "esquema/schema_tabla_raw.json")
    mlflow.log_dict({str(k): int(v) for k, v in conteo_lotes.items()}, "datos/registros_por_lote.json")
    try:
        mlflow.log_input(mlflow.data.load_delta(table_name=TABLA_RAW, version=version_raw,
                                                name="tabla_raw"), context="salida")
    except Exception as e:
        print(f"⚠️ No se pudo registrar el dataset Delta: {e}")

print(f"✅ Run {run.info.run_name}: {HOST}/ml/experiments/{EXPERIMENT_ID}/runs/{run.info.run_id}")

# COMMAND ----------

# CELDA 7: SALIDA PARA EL PIPELINE (task values)
salida = {"registros_extraidos": len(data), "lote_max": LOTE_MAX, "version_tabla_raw": int(version_raw)}
try:
    for clave, valor in salida.items():
        dbutils.jobs.taskValues.set(key=clave, value=valor)
except Exception as e:
    print(f"(Ejecución interactiva: no se publican task values) {e}")
print(json.dumps(salida))