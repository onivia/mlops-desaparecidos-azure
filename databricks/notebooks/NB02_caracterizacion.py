# Databricks notebook source
# MAGIC %md
# MAGIC # NB02_caracterizacion — Caracterización y control de calidad
# MAGIC
# MAGIC EDA de los datos de origen y **puerta de calidad** del pipeline: si alguna validación falla, el notebook falla y el pipeline se detiene.
# MAGIC
# MAGIC **Parámetros de entrada**
# MAGIC - `iteracion` (default 1) y `job_run_id` (opcional): solo para nombrar el run y la trazabilidad.
# MAGIC
# MAGIC **Entrada:** tabla Delta `workspace.mlops_desaparecidos.tabla_raw`.
# MAGIC
# MAGIC **Salida**
# MAGIC - Tabla Delta `workspace.mlops_desaparecidos.tabla_caracterizacion`: **copia exacta** de `tabla_raw` (no se modifica ningún dato).
# MAGIC - Run `caracterizacion_iterNN` en el experimento `MLOps-Desaparecidos/NB02_caracterizacion`, con el EDA como artefactos.
# MAGIC - Task values: `registros`, `calidad_ok`.

# COMMAND ----------

# MAGIC %pip install -q mlflow seaborn

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# CELDA 1: PARÁMETROS DE ENTRADA
dbutils.widgets.text("iteracion", "1", "Iteración")
dbutils.widgets.text("job_run_id", "", "Id de ejecución del job")

# Si el Job envía una iteración inválida, se detiene en lugar de usar la 1.
valor_iteracion = dbutils.widgets.get("iteracion").strip()
try:
    ITERACION = int(valor_iteracion)
except ValueError as e:
    raise ValueError(f"'iteracion' debe ser un entero positivo; recibido: {valor_iteracion!r}") from e
if ITERACION <= 0:
    raise ValueError(f"'iteracion' debe ser mayor que cero; recibido: {ITERACION}")
JOB_RUN_ID = dbutils.widgets.get("job_run_id").strip()

# COMMAND ----------

# CELDA 2: IMPORTS Y CONFIGURACIÓN
import json
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from pyspark.sql import functions as F

warnings.filterwarnings("ignore")

# ⚠️ Si su workspace usa otro catálogo (p. ej. "main"), cámbielo aquí
#    y en los demás notebooks del pipeline.
CATALOGO = "workspace"
ESQUEMA = "mlops_desaparecidos"
TABLA_ENTRADA = f"{CATALOGO}.{ESQUEMA}.tabla_raw"
TABLA_SALIDA = f"{CATALOGO}.{ESQUEMA}.tabla_caracterizacion"

CIUDADES = {"Bogotá": "11001", "Cali": "76001", "Medellín": "5001",
            "Pereira": "66001", "Barranquilla": "8001"}
COLUMNAS_TECNICAS = ["id", "posicion", "lote"]     # identificadores, no son datos del caso
COLUMNAS_REQUERIDAS = [                            # las que usan las etapas siguientes
    "id", "lote", "estado_de_la_desaparicion", "codigo_dane_municipio",
    "a_o_de_la_desaparicion", "fecha_de_la_desaparicion",
]
VALORES_SIN_INFO = {"sin información", "sin informacion", "no sabe / no informa"}

# COMMAND ----------

# CELDA 3: CARGA DE tabla_raw
assert spark.catalog.tableExists(TABLA_ENTRADA), \
    f"No existe {TABLA_ENTRADA}. Ejecute primero NB01_extraccion."

version_entrada = spark.sql(f"DESCRIBE HISTORY {TABLA_ENTRADA} LIMIT 1").first()["version"]
# Se lee la versión exacta: la salida será una copia fiel de ESTA versión
df = spark.read.option("versionAsOf", version_entrada).table(TABLA_ENTRADA)
total = df.count()
# Prevalidaciones antes del EDA: evitan divisiones por cero y columnas inexistentes.
# La puerta de calidad de la CELDA 9 conserva sus comprobaciones originales.
if total == 0:
    raise ValueError("Puerta de calidad NO superada: tabla_raw está vacía.")
columnas_eda = COLUMNAS_REQUERIDAS + ["municipio_donde_ocurre_la_desaparicion_dane"]
faltantes_eda = [c for c in columnas_eda if c not in df.columns]
if faltantes_eda:
    raise ValueError(f"Puerta de calidad NO superada: faltan columnas necesarias: {faltantes_eda}")
columnas_datos = [c for c in df.columns if c not in COLUMNAS_TECNICAS]

print(f"Entrada: {TABLA_ENTRADA} (versión {version_entrada}) | {total} registros | {len(df.columns)} columnas")
display(df.limit(5))

# COMMAND ----------

# CELDA 4: REGISTROS POR CIUDAD Y POR LOTE
codigo_a_ciudad = {v: k for k, v in CIUDADES.items()}
por_ciudad = (df.groupBy("codigo_dane_municipio", "municipio_donde_ocurre_la_desaparicion_dane")
                .count().orderBy(F.desc("count")).toPandas())
por_lote = df.groupBy("lote").count().orderBy("lote").toPandas()

display(por_ciudad)
display(por_lote)

# COMMAND ----------

# CELDA 5: DISTRIBUCIÓN DEL ESTADO (origen de la variable objetivo)
por_estado = df.groupBy("estado_de_la_desaparicion").count().orderBy(F.desc("count")).toPandas()
por_estado["pct"] = (por_estado["count"] / total * 100).round(2)
display(por_estado)

# COMMAND ----------

# CELDA 6: CATEGORÍAS DE LAS COLUMNAS CASI CONSTANTES
# Sustentan la eliminación de columnas en NB03_transformacion.
columnas_categoricas = [
    "clasificacion_de_la_desaparicion",
    "identidad_de_genero_del_desaparecido",
    "orientacion_sexual_del_desaparecido",
    "pais_donde_ocurre_la_desaparicion",
    "contexto",
]
tablas_categorias = {}
for columna in columnas_categoricas:
    if columna not in df.columns:
        continue
    tabla = df.groupBy(columna).count().orderBy(F.desc("count")).toPandas()
    tabla["pct"] = (tabla["count"] / total * 100).round(2)
    tablas_categorias[columna] = tabla
    print(f"── {columna}: {len(tabla)} categorías; la principal concentra {tabla['pct'].iloc[0]}%")
    display(tabla)

# COMMAND ----------

# CELDA 7: NULOS, "SIN INFORMACIÓN" Y DUPLICADOS
# ── Nulos por columna ──
fila_nulos = df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]).first()
nulos = {c: int(fila_nulos[c] or 0) for c in df.columns}
total_nulos = sum(nulos.values())

# ── Porcentaje de valores "Sin información" / "No Sabe / No Informa" ──
fila_sin_info = df.select([
    F.sum(F.lower(F.trim(F.col(c))).isin(list(VALORES_SIN_INFO)).cast("int")).alias(c)
    for c in columnas_datos
]).first()
pct_sin_info = {c: round((fila_sin_info[c] or 0) / total * 100, 2) for c in columnas_datos}
pct_sin_info_global = round(sum(fila_sin_info[c] or 0 for c in columnas_datos)
                            / (total * len(columnas_datos)) * 100, 2)

# ── Duplicados de contenido (se ignoran los identificadores técnicos) ──
filas_unicas = df.select(columnas_datos).distinct().count()
duplicados = total - filas_unicas
pct_duplicados = round(duplicados / total * 100, 2)

calidad = pd.DataFrame({
    "columna": columnas_datos,
    "nulos": [nulos[c] for c in columnas_datos],
    "pct_sin_informacion": [pct_sin_info[c] for c in columnas_datos],
}).sort_values("pct_sin_informacion", ascending=False)

print(f"Nulos totales: {total_nulos} | Duplicados de contenido: {duplicados} ({pct_duplicados}%) "
      f"| % 'Sin información' global: {pct_sin_info_global}%")
display(calidad)

# COMMAND ----------

# CELDA 8: GRÁFICA — REGISTROS POR AÑO Y CIUDAD
pdf_anio = (df.groupBy("a_o_de_la_desaparicion", "municipio_donde_ocurre_la_desaparicion_dane")
              .count().toPandas()
              .rename(columns={"a_o_de_la_desaparicion": "año",
                               "municipio_donde_ocurre_la_desaparicion_dane": "municipio",
                               "count": "registros"}))
pdf_anio["año"] = pd.to_numeric(pdf_anio["año"], errors="coerce")
registros_anio_sin_info = int(pdf_anio.loc[pdf_anio["año"].isna(), "registros"].sum())
pdf_anio = pdf_anio.dropna(subset=["año"]).astype({"año": int}).sort_values("año")

fig_anio, ax = plt.subplots(figsize=(14, 6))
if pdf_anio.empty:
    ax.text(0.5, 0.5, "Sin años válidos en la extracción actual", ha="center", va="center")
    ax.axis("off")
else:
    sns.barplot(data=pdf_anio, x="año", y="registros", hue="municipio", ax=ax)
    ax.set_title(f"Registros por año y municipio — iteración {ITERACION}", fontweight="bold")
    ax.set_xlabel("Año de la desaparición")
    ax.set_ylabel("Registros")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Municipio", bbox_to_anchor=(1.02, 1), loc="upper left")
fig_anio.tight_layout()
plt.show()

# COMMAND ----------

# CELDA 9: PUERTA DE CALIDAD
# Se reúnen TODOS los fallos y se reportan juntos. Un fallo detiene el pipeline.
fallos, avisos = [], []

if total == 0:
    fallos.append("La tabla está vacía.")
faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
if faltantes:
    fallos.append(f"Faltan columnas requeridas: {faltantes}")
if nulos.get("id", 0) > 0:
    fallos.append(f"Hay {nulos['id']} registros sin 'id'.")
if df.select("id").distinct().count() != total:
    fallos.append("La columna 'id' tiene valores repetidos.")
codigos_presentes = {r[0] for r in df.select("codigo_dane_municipio").distinct().collect()}
fuera = codigos_presentes - set(CIUDADES.values())
if fuera:
    fallos.append(f"Hay códigos DANE fuera del estudio: {fuera}")
if len(codigos_presentes) < len(CIUDADES):
    avisos.append(f"Solo hay {len(codigos_presentes)} de {len(CIUDADES)} ciudades en la extracción.")
if duplicados > 0:
    avisos.append(f"{duplicados} registros con contenido duplicado (ids distintos).")

for aviso in avisos:
    print(f"⚠️ {aviso}")
CALIDAD_OK = not fallos
assert CALIDAD_OK, "Puerta de calidad NO superada:\n- " + "\n- ".join(fallos)
print(f"✅ Puerta de calidad superada: {total} registros, {len(codigos_presentes)} ciudades, id único.")

# COMMAND ----------

# CELDA 10: GUARDAR tabla_caracterizacion (COPIA EXACTA DE tabla_raw)
(df.write
   .format("delta")
   .mode("overwrite")
   .option("overwriteSchema", "true")
   .saveAsTable(TABLA_SALIDA))

spark.sql(f"""
    COMMENT ON TABLE {TABLA_SALIDA} IS
    'Copia validada de {TABLA_ENTRADA} v{version_entrada} generada por NB02_caracterizacion (iteración {ITERACION})'
""")

version_salida = spark.sql(f"DESCRIBE HISTORY {TABLA_SALIDA} LIMIT 1").first()["version"]
df_salida = spark.table(TABLA_SALIDA)
assert df_salida.count() == total and df_salida.columns == df.columns, \
    "La tabla de salida no es una copia exacta de la entrada."
print(f"✅ {TABLA_SALIDA} | versión {version_salida} | {total} registros (sin modificaciones)")

# COMMAND ----------

# CELDA 11: REGISTRO EN MLFLOW
import re
from databricks.sdk import WorkspaceClient


def contexto_notebook(nombre_por_defecto):
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


RUTA_NOTEBOOK, NOMBRE_NOTEBOOK, HOST = contexto_notebook("NB02_caracterizacion")
USUARIO = spark.sql("SELECT current_user()").first()[0]
CARPETA_EXPERIMENTOS = f"/Users/{USUARIO}/MLOps-Desaparecidos"
WorkspaceClient().workspace.mkdirs(CARPETA_EXPERIMENTOS)

mlflow.set_registry_uri("databricks-uc")
EXPERIMENT_ID = mlflow.set_experiment(f"{CARPETA_EXPERIMENTOS}/{NOMBRE_NOTEBOOK}").experiment_id
mlflow.set_experiment_tag("mlflow.experimentKind", "custom_model_development")
mlflow.end_run()

anios_validos = pdf_anio["año"] if not pdf_anio.empty else pd.Series([0])

with mlflow.start_run(run_name=f"caracterizacion_iter{ITERACION:02d}", experiment_id=EXPERIMENT_ID,
                      description=f"EDA y control de calidad de `{TABLA_ENTRADA}` v{version_entrada}") as run:
    mlflow.set_tags({
        "etapa_pipeline": "02-caracterizacion",
        "iteracion": str(ITERACION),
        "job_run_id": JOB_RUN_ID,
        "notebook": RUTA_NOTEBOOK or NOMBRE_NOTEBOOK,
        "tabla_entrada": TABLA_ENTRADA,
        "tabla_entrada_version": str(version_entrada),
        "tabla_salida": TABLA_SALIDA,
        "tabla_salida_version": str(version_salida),
        "calidad_ok": str(CALIDAD_OK),
    })
    mlflow.log_params({"iteracion": ITERACION, "tabla_entrada": TABLA_ENTRADA,
                       "tabla_entrada_version": version_entrada, "tabla_salida": TABLA_SALIDA,
                       "tabla_salida_version": version_salida})
    mlflow.log_metrics({
        "n_registros": total,
        "n_columnas": len(df.columns),
        "n_lotes": len(por_lote),
        "n_ciudades": len(codigos_presentes),
        "total_nulos": total_nulos,
        "registros_duplicados": duplicados,
        "pct_duplicados": pct_duplicados,
        "pct_sin_informacion_global": pct_sin_info_global,
        "registros_anio_sin_informacion": registros_anio_sin_info,
        "anio_minimo": int(anios_validos.min()),
        "anio_maximo": int(anios_validos.max()),
        "n_avisos_calidad": len(avisos),
    })
    for _, fila in por_ciudad.iterrows():
        ciudad = codigo_a_ciudad.get(fila["codigo_dane_municipio"], fila["codigo_dane_municipio"])
        mlflow.log_metric(f"registros_{nombre_metrica(ciudad)}", int(fila["count"]))
    for _, fila in por_estado.iterrows():
        mlflow.log_metric(f"pct_estado_{nombre_metrica(fila['estado_de_la_desaparicion'])}", float(fila["pct"]))

    mlflow.log_figure(fig_anio, "graficas/registros_por_anio_y_ciudad.png")
    mlflow.log_text(por_ciudad.to_csv(index=False), "eda/registros_por_ciudad.csv")
    mlflow.log_text(por_lote.to_csv(index=False), "eda/registros_por_lote.csv")
    mlflow.log_text(por_estado.to_csv(index=False), "eda/distribucion_estado.csv")
    mlflow.log_text(pdf_anio.to_csv(index=False), "eda/registros_por_anio_y_ciudad.csv")
    mlflow.log_text(calidad.to_csv(index=False), "calidad/nulos_y_sin_informacion.csv")
    for columna, tabla in tablas_categorias.items():
        mlflow.log_text(tabla.to_csv(index=False), f"eda/categorias_{columna}.csv")
    mlflow.log_dict({"fallos": fallos, "avisos": avisos}, "calidad/resultado_validaciones.json")
    mlflow.log_dict({c: t for c, t in df.dtypes}, "esquema/schema.json")
    try:
        mlflow.log_input(mlflow.data.load_delta(table_name=TABLA_ENTRADA, version=version_entrada,
                                                name="tabla_raw"), context="entrada")
        mlflow.log_input(mlflow.data.load_delta(table_name=TABLA_SALIDA, version=version_salida,
                                                name="tabla_caracterizacion"), context="salida")
    except Exception as e:
        print(f"⚠️ No se pudo registrar el dataset Delta: {e}")

print(f"✅ Run {run.info.run_name}: {HOST}/ml/experiments/{EXPERIMENT_ID}/runs/{run.info.run_id}")

# COMMAND ----------

# CELDA 12: SALIDA PARA EL PIPELINE (task values)
salida = {"registros": total, "calidad_ok": CALIDAD_OK, "version_tabla_caracterizacion": int(version_salida)}
try:
    for clave, valor in salida.items():
        dbutils.jobs.taskValues.set(key=clave, value=valor)
except Exception as e:
    print(f"(Ejecución interactiva: no se publican task values) {e}")
print(json.dumps(salida))