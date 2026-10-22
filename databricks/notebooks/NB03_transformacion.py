# Databricks notebook source
# MAGIC %md
# MAGIC # NB03_transformacion — Limpieza y transformación de datos
# MAGIC
# MAGIC **Parámetros de entrada**
# MAGIC - `anio_min` (default 2006) y `anio_max` (default 2026): rango de años que se conserva.
# MAGIC - `iteracion` (default 1) y `job_run_id` (opcional): solo para nombrar el run y la trazabilidad.
# MAGIC
# MAGIC **Entrada:** tabla Delta `workspace.mlops_desaparecidos.tabla_caracterizacion`.
# MAGIC
# MAGIC **Función**
# MAGIC 1. Eliminar columnas redundantes o sin valor predictivo.
# MAGIC 2. `id` → `id_registro` (id estable del API) y nombres cortos para el resto.
# MAGIC 3. Tipos: `Fecha` a date, `Año` a texto.
# MAGIC 4. Filtrar los años `anio_min`–`anio_max`.
# MAGIC 5. Crear la variable objetivo **`localizada`**: 1 si apareció vivo o muerto, 0 si sigue desaparecido.
# MAGIC
# MAGIC **Salida**
# MAGIC - Tabla Delta `workspace.mlops_desaparecidos.tabla_transformacion` (sobrescrita).
# MAGIC - Run `transformacion_iterNN` en el experimento `MLOps-Desaparecidos/NB03_transformacion`.
# MAGIC - Task values: `registros`, `pct_localizada`.

# COMMAND ----------

# MAGIC %pip install -q mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# CELDA 1: PARÁMETROS DE ENTRADA
dbutils.widgets.text("anio_min", "2006", "Año mínimo")
dbutils.widgets.text("anio_max", "2026", "Año máximo")
dbutils.widgets.text("iteracion", "1", "Iteración")
# Desde el Job Hijo se debe asignar job_run_id = {{job.run_id}}.
dbutils.widgets.text("job_run_id", "", "Id de ejecución del job")


def leer_entero(nombre):
    """Exige un valor entero válido; no sustituye errores del Job por defaults."""
    valor = dbutils.widgets.get(nombre).strip()
    try:
        return int(valor)
    except ValueError as e:
        raise ValueError(f"Parámetro '{nombre}' inválido: {valor!r}. Debe ser un entero.") from e


ANIO_MIN = leer_entero("anio_min")
ANIO_MAX = leer_entero("anio_max")
ITERACION = leer_entero("iteracion")
if ITERACION <= 0:
    raise ValueError(f"Parámetro 'iteracion' inválido: {ITERACION}. Debe ser mayor que cero.")
JOB_RUN_ID = dbutils.widgets.get("job_run_id").strip()
assert ANIO_MIN <= ANIO_MAX, "anio_min debe ser menor o igual que anio_max."


# COMMAND ----------

# CELDA 2: IMPORTS Y CONFIGURACIÓN
import json
import warnings
import matplotlib.pyplot as plt
import mlflow
from pyspark.sql import functions as F

warnings.filterwarnings("ignore")

# ⚠️ Si su workspace usa otro catálogo (p. ej. "main"), cámbielo aquí
#    y en los demás notebooks del pipeline.
CATALOGO = "workspace"
ESQUEMA = "mlops_desaparecidos"
TABLA_ENTRADA = f"{CATALOGO}.{ESQUEMA}.tabla_caracterizacion"
TABLA_SALIDA = f"{CATALOGO}.{ESQUEMA}.tabla_transformacion"
TARGET = "localizada"

# COMMAND ----------

# CELDA 3: CARGA DE tabla_caracterizacion
assert spark.catalog.tableExists(TABLA_ENTRADA), \
    f"No existe {TABLA_ENTRADA}. Ejecute primero NB02_caracterizacion."

version_entrada = spark.sql(f"DESCRIBE HISTORY {TABLA_ENTRADA} LIMIT 1").first()["version"]
df_entrada = spark.read.option("versionAsOf", version_entrada).table(TABLA_ENTRADA)
total = df_entrada.count()
assert total > 0, "La tabla de entrada está vacía."

print(f"Entrada: {TABLA_ENTRADA} (versión {version_entrada}) | {total} registros | {len(df_entrada.columns)} columnas")

# COMMAND ----------

# CELDA 4: ELIMINACIÓN DE COLUMNAS
# Redundantes con otra columna, casi constantes o sin valor predictivo.
columnas_a_eliminar = [
    "ciclo_vital_del_desaparecido",                    # redundante con el rango de edad
    "grupo_de_edad_quinquenal_del_desaparecido",       # redundante con el rango de edad
    "grupo_mayor_y_menor_de_edad_del_desaparecido",    # redundante con el rango de edad
    "clasificacion_de_la_desaparicion",                # ~98% "Sin información"
    "identidad_de_genero_del_desaparecido",            # ~99% "No Sabe / No Informa"
    "orientacion_sexual_del_desaparecido",             # ~99% "No Sabe / No Informa"
    "transgenero",                                     # ~99% "No Sabe / No Informa"
    "pueblo_indigena_del_desaparecido",                # casi constante
    "pais_donde_ocurre_la_desaparicion",               # constante (Colombia)
    "zona_donde_ocurre_la_desaparicion",               # poca variación
    "codigo_dane_departamento",                        # redundante con el nombre
    "codigo_dane_municipio",                           # redundante con el nombre
    "departamento_donde_ocurre_la_desaparicion_dane",  # redundante con la ciudad
    "contexto",                                        # constante
]
inexistentes = [c for c in columnas_a_eliminar if c not in df_entrada.columns]
assert not inexistentes, f"Columnas no encontradas: {inexistentes}"

df = df_entrada.drop(*columnas_a_eliminar)
print(f"Columnas eliminadas: {len(columnas_a_eliminar)} | restantes: {len(df.columns)}")

# COMMAND ----------

# CELDA 5: RENOMBRAMIENTO DE COLUMNAS
renombres = {
    "id": "id_registro",            # id estable del API: identifica el registro en todas las iteraciones
    "entidad_que_realiza_el_registro_de_la_desaparicion": "RegistradoPor",
    "estado_de_la_desaparicion": "Estado",
    "sexo_del_desaparecido": "Sexo",
    "grupo_de_edad_judicial_del_desaparecido": "RangoEdad",
    "estado_civil_del_desaparecido": "EstadoCivil",
    "escolaridad_del_desaparecido": "Escolaridad",
    "pertenencia_etnica_del_desaparecido": "Etnia",
    "pertenencia_grupal": "PertenenciaGrupal",
    "pais_de_nacimiento_del_desaparecido": "PaisNacimiento",
    "fecha_de_la_desaparicion": "Fecha",
    "a_o_de_la_desaparicion": "Año",
    "mes_de_la_desaparicion": "Mes",
    "dia_de_la_desaparicion": "Dia",
    "municipio_donde_ocurre_la_desaparicion_dane": "Ciudad",
    "localidad_donde_ocurre_la_desaparicion": "Localidad",
}
inexistentes = [c for c in renombres if c not in df.columns]
assert not inexistentes, f"Columnas no encontradas: {inexistentes}"

for original, nuevo in renombres.items():
    df = df.withColumnRenamed(original, nuevo)
print(f"Columnas actuales: {df.columns}")

# COMMAND ----------

# CELDA 6: TIPOS DE DATOS
# Fecha viene como "2006-12-29T00:00:00.000": se toma la parte de la fecha.
df = (df
      .withColumn("Año", F.col("Año").cast("string"))
      .withColumn("Fecha", F.to_date(F.substring(F.col("Fecha"), 1, 10), "yyyy-MM-dd")))

fechas_nulas_total = df.filter(F.col("Fecha").isNull()).count()
print(f"Fechas no convertibles (antes del filtro de años): {fechas_nulas_total}")

# COMMAND ----------

# CELDA 7: FILTRO POR AÑO
# try_cast convierte en NULL los valores no numéricos ("Sin información")
# y el filtro los descarta.
anio_int = F.expr("try_cast(`Año` as int)")
df = df.filter(anio_int.between(ANIO_MIN, ANIO_MAX))
registros_tras_filtro = df.count()
assert registros_tras_filtro > 0, (
    f"No quedan registros con año entre {ANIO_MIN} y {ANIO_MAX}. Aumente el límite de extracción.")
print(f"Registros con año {ANIO_MIN}-{ANIO_MAX}: {registros_tras_filtro} (excluidos: {total - registros_tras_filtro})")

# COMMAND ----------

# CELDA 8: VARIABLE OBJETIVO — localizada
#   1 = apareció vivo o apareció muerto
#   0 = sigue desaparecido
# Se normaliza el texto (minúsculas, sin tildes) para que variantes como
# "Apareció vivo" / "Aparecio vivo" coincidan.
estado_normalizado = F.lower(F.trim(F.translate(F.col("Estado"), "áéíóúÁÉÍÓÚ", "aeiouAEIOU")))

df = df.withColumn(
    TARGET,
    F.when(estado_normalizado.isin("aparecio vivo", "aparecio muerto"), F.lit(1))
     .when(estado_normalizado == "desaparecido", F.lit(0))
     .otherwise(F.lit(None).cast("int"))
)

no_mapeados = df.filter(F.col(TARGET).isNull()).groupBy("Estado").count().toPandas()
if len(no_mapeados) > 0:
    print("⚠️ Estados excluidos (no corresponden a ninguna clase):")
    display(no_mapeados)

df_salida = df.filter(F.col(TARGET).isNotNull())
n_registros = df_salida.count()
assert n_registros > 0, "No quedan registros con un estado válido."

distribucion = df_salida.groupBy(TARGET).count().orderBy(TARGET).toPandas()
distribucion["pct"] = (distribucion["count"] / n_registros * 100).round(2)
pct_localizada = float(distribucion.loc[distribucion[TARGET] == 1, "pct"].sum())
display(distribucion)

# COMMAND ----------

# CELDA 9: ORDEN DE COLUMNAS Y VALIDACIONES
columnas_finales = ["id_registro", "lote", "posicion", "Fecha", "Año", "Estado", TARGET,
                    "Ciudad", "Localidad", "Sexo", "RangoEdad", "Escolaridad", "EstadoCivil",
                    "Etnia", "PaisNacimiento", "PertenenciaGrupal", "RegistradoPor", "Mes", "Dia"]
faltantes = [c for c in columnas_finales if c not in df_salida.columns]
assert not faltantes, f"Faltan columnas en el resultado: {faltantes}"
df_salida = df_salida.select(*columnas_finales)

assert df_salida.select("id_registro").distinct().count() == n_registros, "id_registro debe ser único."
assert df_salida.filter(F.col("Fecha").isNull()).count() == 0, "Hay fechas nulas en el resultado."

por_lote = df_salida.groupBy("lote").count().orderBy("lote").toPandas()
por_anio = df_salida.groupBy("Año").count().orderBy("Año").toPandas()
print("✅ Validaciones superadas")
display(por_lote)

# COMMAND ----------

# CELDA 10: GUARDAR tabla_transformacion (SOBRESCRITURA)
(df_salida.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLA_SALIDA))

spark.sql(f"""
    COMMENT ON TABLE {TABLA_SALIDA} IS
    'Datos limpios y transformados (años {ANIO_MIN}-{ANIO_MAX}, con objetivo {TARGET}) generados por NB03_transformacion a partir de {TABLA_ENTRADA} v{version_entrada}'
""")

version_salida = spark.sql(f"DESCRIBE HISTORY {TABLA_SALIDA} LIMIT 1").first()["version"]
assert spark.table(TABLA_SALIDA).count() == n_registros, "La tabla guardada no tiene los registros esperados."
print(f"✅ {TABLA_SALIDA} | versión {version_salida} | {n_registros} registros | {len(columnas_finales)} columnas")
display(spark.table(TABLA_SALIDA).limit(5))

# COMMAND ----------

# CELDA 11: REGISTRO EN MLFLOW
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


RUTA_NOTEBOOK, NOMBRE_NOTEBOOK, HOST = contexto_notebook("NB03_transformacion")
USUARIO = spark.sql("SELECT current_user()").first()[0]
CARPETA_EXPERIMENTOS = f"/Users/{USUARIO}/MLOps-Desaparecidos"
WorkspaceClient().workspace.mkdirs(CARPETA_EXPERIMENTOS)

mlflow.set_registry_uri("databricks-uc")
EXPERIMENT_ID = mlflow.set_experiment(f"{CARPETA_EXPERIMENTOS}/{NOMBRE_NOTEBOOK}").experiment_id
mlflow.set_experiment_tag("mlflow.experimentKind", "custom_model_development")
mlflow.end_run()

with mlflow.start_run(run_name=f"transformacion_iter{ITERACION:02d}", experiment_id=EXPERIMENT_ID,
                      description=f"`{TABLA_ENTRADA}` v{version_entrada} → `{TABLA_SALIDA}` v{version_salida}") as run:
    mlflow.set_tags({
        "etapa_pipeline": "03-transformacion",
        "iteracion": str(ITERACION),
        "job_run_id": JOB_RUN_ID,
        "notebook": RUTA_NOTEBOOK or NOMBRE_NOTEBOOK,
        "tabla_entrada": TABLA_ENTRADA,
        "tabla_entrada_version": str(version_entrada),
        "tabla_salida": TABLA_SALIDA,
        "tabla_salida_version": str(version_salida),
    })
    mlflow.log_params({
        "iteracion": ITERACION,
        "anio_min": ANIO_MIN,
        "anio_max": ANIO_MAX,
        "target": TARGET,
        "definicion_target": "1 = apareció vivo o muerto; 0 = desaparecido",
        "n_columnas_eliminadas": len(columnas_a_eliminar),
        "n_columnas_renombradas": len(renombres),
        "tabla_entrada_version": version_entrada,
        "tabla_salida_version": version_salida,
    })
    mlflow.log_metrics({
        "registros_entrada": total,
        "registros_tras_filtro_anio": registros_tras_filtro,
        "registros_estado_no_mapeado": registros_tras_filtro - n_registros,
        "registros_salida": n_registros,
        "pct_excluidos": round((total - n_registros) / total * 100, 2),
        "pct_localizada": pct_localizada,
        "n_columnas_entrada": len(df_entrada.columns),
        "n_columnas_salida": len(columnas_finales),
        "fechas_no_convertibles": fechas_nulas_total,
        "n_lotes": len(por_lote),
    })

    mlflow.log_dict({"columnas_eliminadas": columnas_a_eliminar}, "transformaciones/columnas_eliminadas.json")
    mlflow.log_dict(renombres, "transformaciones/renombres.json")
    mlflow.log_dict({c: t for c, t in df_entrada.dtypes}, "esquema/schema_entrada.json")
    mlflow.log_dict({c: t for c, t in df_salida.dtypes}, "esquema/schema_salida.json")
    mlflow.log_text(distribucion.to_csv(index=False), "datos/distribucion_target.csv")
    mlflow.log_text(por_lote.to_csv(index=False), "datos/registros_por_lote.csv")
    mlflow.log_text(por_anio.to_csv(index=False), "datos/registros_por_anio.csv")
    if len(no_mapeados) > 0:
        mlflow.log_text(no_mapeados.to_csv(index=False), "datos/estados_no_mapeados.csv")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(por_anio["Año"], por_anio["count"], color="#2a6f97")
    ax.set_title(f"Registros por año — tabla_transformacion (iteración {ITERACION})", fontweight="bold")
    ax.set_xlabel("Año")
    ax.set_ylabel("Registros")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    mlflow.log_figure(fig, "graficas/registros_por_anio.png")
    plt.show()

    try:
        mlflow.log_input(mlflow.data.load_delta(table_name=TABLA_ENTRADA, version=version_entrada,
                                                name="tabla_caracterizacion"), context="entrada")
        mlflow.log_input(mlflow.data.load_delta(table_name=TABLA_SALIDA, version=version_salida,
                                                name="tabla_transformacion"), context="salida")
    except Exception as e:
        print(f"⚠️ No se pudo registrar el dataset Delta: {e}")

print(f"✅ Run {run.info.run_name}: {HOST}/ml/experiments/{EXPERIMENT_ID}/runs/{run.info.run_id}")

# COMMAND ----------

# CELDA 12: SALIDA PARA EL PIPELINE (task values)
salida = {"registros": n_registros, "pct_localizada": pct_localizada,
          "version_tabla_transformacion": int(version_salida)}
try:
    for clave, valor in salida.items():
        dbutils.jobs.taskValues.set(key=clave, value=valor)
except Exception as e:
    print(f"(Ejecución interactiva: no se publican task values) {e}")
print(json.dumps(salida))