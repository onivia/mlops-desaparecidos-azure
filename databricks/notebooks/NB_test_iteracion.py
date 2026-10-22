# Databricks notebook source
# MAGIC %md
# MAGIC # NB_test_iteracion — Prueba de orquestación del Job Padre
# MAGIC
# MAGIC **Propósito:** reemplazar temporalmente la Task `Run Job` dentro del `For each` del Job Padre `pipeline_mlops_desaparecidos`.
# MAGIC
# MAGIC - Recibe los parámetros reales enviados por `NB00_configuracion` a través del `For each`.
# MAGIC - Verifica que el número de iteración y la cantidad de registros sean coherentes.
# MAGIC - Muestra un escenario **100 % SIMULADO** de Challenger, Champion, métricas, promoción y monitoreo.
# MAGIC - **No** consulta la API, **no** entrena modelos, **no** registra modelos y **no** lee ni modifica tablas Delta.
# MAGIC
# MAGIC > Para probar solo la primera iteración, ejecuta el Job Padre con `max_iteraciones=1` y `reiniciar_modelos=false`. El NB00 sigue generando su propio run de configuración en MLflow.

# COMMAND ----------

# CELDA 1. Parámetros del elemento actual del For each
# Los valores por defecto solo se usan al ejecutar este notebook manualmente.
dbutils.widgets.text("iteracion", "1", "Iteración actual")
dbutils.widgets.text("limite_registros", "1000", "Registros acumulados")
dbutils.widgets.text("tamano_lote", "1000", "Tamaño del lote")

ITERACION = int(dbutils.widgets.get("iteracion").strip())
LIMITE_REGISTROS = int(dbutils.widgets.get("limite_registros").strip())
TAMANO_LOTE = int(dbutils.widgets.get("tamano_lote").strip())

if ITERACION < 1 or TAMANO_LOTE < 1:
    raise ValueError("iteracion y tamano_lote deben ser mayores que cero.")

limite_esperado = ITERACION * TAMANO_LOTE
if LIMITE_REGISTROS != limite_esperado:
    raise ValueError(
        f"Parámetros incoherentes: limite_registros={LIMITE_REGISTROS}; "
        f"se esperaba {limite_esperado} para iteracion={ITERACION}."
    )

print("✅ PARÁMETROS DEL JOB PADRE RECIBIDOS CORRECTAMENTE")
print(f"Iteración actual         : {ITERACION}")
print(f"Tamaño del lote          : {TAMANO_LOTE:,}")
print(f"Registros acumulados     : {LIMITE_REGISTROS:,}")
print(f"Registros de lotes previos: {LIMITE_REGISTROS - TAMANO_LOTE:,}")
print(f"Registros del lote actual: {TAMANO_LOTE:,}")

# COMMAND ----------

# CELDA 2. Escenarios de DEMOSTRACIÓN: no representan modelos ni métricas reales.
# Se definen de antemano para ilustrar cómo se verían cinco iteraciones.
ESCENARIOS = {
    1: dict(challenger="Random Forest", auc_ch=0.730, f1_ch=0.690,
            champion_anterior="No existe", auc_ant=None, f1_ant=None,
            promover=True, champion_final="Random Forest",
            drift="No aplica: primera iteración"),
    2: dict(challenger="Regresión Logística", auc_ch=0.700, f1_ch=0.660,
            champion_anterior="Random Forest", auc_ant=0.740, f1_ant=0.700,
            promover=False, champion_final="Random Forest",
            drift="Sin drift (simulado)"),
    3: dict(challenger="XGBoost", auc_ch=0.790, f1_ch=0.750,
            champion_anterior="Random Forest", auc_ant=0.710, f1_ant=0.680,
            promover=True, champion_final="XGBoost",
            drift="Drift significativo (simulado)"),
    4: dict(challenger="LightGBM", auc_ch=0.770, f1_ch=0.730,
            champion_anterior="XGBoost", auc_ant=0.780, f1_ant=0.740,
            promover=False, champion_final="XGBoost",
            drift="Sin drift (simulado)"),
    5: dict(challenger="HistGradientBoosting", auc_ch=0.800, f1_ch=0.760,
            champion_anterior="XGBoost", auc_ant=0.780, f1_ant=0.750,
            promover=True, champion_final="HistGradientBoosting",
            drift="Drift moderado (simulado)"),
}

escenario = ESCENARIOS.get(ITERACION)

def formato_metrica(valor):
    return "No aplica" if valor is None else f"{valor:.3f}"

print("\n" + "=" * 58)
print("🧪 RESULTADOS TOTALMENTE SIMULADOS — NO SE ENTRENAN MODELOS")
print("=" * 58)

if escenario is None:
    print(f"No hay escenario ficticio predefinido para la iteración {ITERACION}.")
    print("La recepción y validación de parámetros SÍ se completó correctamente.")
else:
    print(f"Modelo Challenger        : {escenario['challenger']} (simulado)")
    print(f"AUC Challenger           : {formato_metrica(escenario['auc_ch'])}")
    print(f"F1 Challenger            : {formato_metrica(escenario['f1_ch'])}")
    print(f"Champion al iniciar      : {escenario['champion_anterior']} (simulado)")
    print(f"AUC Champion anterior    : {formato_metrica(escenario['auc_ant'])}")
    print(f"F1 Champion anterior     : {formato_metrica(escenario['f1_ant'])}")
    print(f"¿Promover Challenger?    : {'Sí' if escenario['promover'] else 'No'}")
    print(f"Champion al finalizar    : {escenario['champion_final']} (simulado)")
    print(f"¿Llamar a Monitoreo?     : {'No' if ITERACION == 1 else 'Sí'}")
    print(f"Resultado drift          : {escenario['drift']}")

print("\n✅ Prueba de la iteración completada.")
print("ℹ️ Sin lectura/escritura de datos Delta, sin entrenamiento y sin registro de modelos.")