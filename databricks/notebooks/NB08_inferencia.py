# Databricks notebook source
# MAGIC %pip install -q mlflow xgboost lightgbm

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import sys
import json

# 1. Ubicación del módulo compartido
RUTA = "/Workspace/MLOPS2026/TallerGeneral/mlops-desaparecidos-api-01"
if RUTA not in sys.path:
    sys.path.insert(0, RUTA)

# 2. Importar la lógica de inferencia
from inferencia import predecir, VARIABLES

# 3. Tomar una fila real de prueba
datos = (
    spark.table("workspace.mlops_desaparecidos.tabla_test")
    .select(*VARIABLES)
    .limit(1)
    .toPandas()
    .iloc[0]
    .to_dict()
)

# 4. Ejecutar la inferencia
resultado = predecir(datos)

# 5. Mostrar el resultado
print(json.dumps(resultado, indent=2, ensure_ascii=False))

# COMMAND ----------

from pyspark.sql import functions as F

tabla = spark.table(
    "workspace.mlops_desaparecidos.tabla_baseline"
)

datos = {
    "Ciudad": "Bogotá",
    "Localidad": "Suba",
    "Sexo": "Masculino",
    "RangoEdad": "Adulto",
    "Escolaridad": "Secundaria",
    "EstadoCivil": "Soltero",
    "Etnia": "Sin información",
    "PaisNacimiento": "Colombia",
    "PertenenciaGrupal": "Sin información",
    "RegistradoPor": "Sin información",
    "Mes": 10,
    "Dia": 10
}

resultados = []

for variable, valor in datos.items():
    columna = F.coalesce(
        F.col(variable).cast("string"),
        F.lit("Sin información")
    )

    frecuencia = tabla.filter(
        columna == str(valor)
    ).count()

    ejemplos = (
        tabla.groupBy(columna.alias("categoria"))
        .count()
        .orderBy(F.desc("count"))
        .limit(3)
        .collect()
    )

    resultados.append({
        "Variable": variable,
        "Valor enviado": valor,
        "Coincidencias": frecuencia,
        "Categorías frecuentes": ", ".join(
            str(f["categoria"]) for f in ejemplos
        )
    })

display(spark.createDataFrame(resultados))

# COMMAND ----------

import json

datos = (
    spark.table("workspace.mlops_desaparecidos.tabla_test")
    .select(*VARIABLES)
    .limit(1)
    .collect()[0]
    .asDict()
)

datos = {
    k: str(v) if v is not None else "Sin información"
    for k, v in datos.items()
}

print(json.dumps(datos, indent=2, ensure_ascii=False))

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd

mlflow.set_registry_uri("databricks-uc")

modelo = mlflow.sklearn.load_model(
    "models:/workspace.mlops_desaparecidos.modelo_desaparecidos@champion"
)

encoder = modelo.named_steps["preprocesamiento"].named_transformers_["categoricas"]

resultado = []

for variable, categorias in zip(VARIABLES, encoder.categories_):
    resultado.append({
        "Variable": variable,
        "Total categorías": len(categorias),
        "Ejemplos": ", ".join(map(str, categorias[:8]))
    })

display(pd.DataFrame(resultado))