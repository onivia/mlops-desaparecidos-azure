# Databricks notebook source
# MAGIC %md
# MAGIC # NB07 — Resultados MLOps para API REST
# MAGIC
# MAGIC Este notebook **consulta** una ejecución existente de `JobPadre01` y construye un JSON consolidado. **No ejecuta de nuevo el pipeline ni modifica las tablas o los modelos.**
# MAGIC
# MAGIC **Uso:** cambia únicamente el widget `run_id_padre` de la primera celda y ejecuta **Run all**. El notebook encuentra automáticamente los Jobs hijos del `For each`, recupera los parámetros reales, consulta Delta y agrega las fechas, duraciones y posibles errores de Databricks Jobs.
# MAGIC
# MAGIC - Fuente de métricas: `tabla_resultados_entrenamiento`, `tabla_promocion` y `tabla_monitoreo`.
# MAGIC - Fuente de metadatos: Databricks Jobs API.
# MAGIC - Tiempos mostrados en `America/Bogota`.
# MAGIC - Si una ejecución no ha terminado, los datos aún no generados pueden aparecer como `null` y el drift como `NO_EVALUADO`.
# MAGIC

# COMMAND ----------

import json
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from databricks.sdk import WorkspaceClient
from pyspark.sql import functions as F

BASE = "workspace.mlops_desaparecidos"
ZONA = ZoneInfo("America/Bogota")
w = WorkspaceClient()

# ÚNICO DATO A CAMBIAR para consultar otra corrida del JobPadre01.
# También se puede pasar run_id_padre como parámetro al ejecutar este notebook.
dbutils.widgets.text("run_id_padre", "535410297498074", "Run ID de JobPadre01")
RUN_PADRE = dbutils.widgets.get("run_id_padre").strip()
if not RUN_PADRE.isdigit():
    raise ValueError("run_id_padre debe ser un identificador numérico")

print("Consultando JobPadre01, run_id:", RUN_PADRE)


# COMMAND ----------

def estado_job(datos):
    state = datos.get("state") or {}
    resultado = state.get("result_state")
    ciclo = state.get("life_cycle_state")
    return {"SUCCESS": "SUCCEEDED"}.get(resultado, resultado or ciclo or "UNKNOWN")


def fecha_local(ms):
    if ms is None or ms == 0:
        return None
    return datetime.fromtimestamp(ms / 1000, timezone.utc).astimezone(
        ZONA
    ).isoformat(timespec="seconds")


def duracion_job(datos):
    inicio, fin = datos.get("start_time"), datos.get("end_time")
    if inicio is None or fin is None:
        return None
    segundos = max(0, (fin - inicio) // 1000)
    return f"{segundos // 60}m{segundos % 60}s"


def datos_tiempo(datos):
    return {
        "estado": estado_job(datos),
        "fecha_inicio": fecha_local(datos.get("start_time")),
        "fecha_fin": fecha_local(datos.get("end_time")),
        "duracion": duracion_job(datos),
    }


def parametros_job(datos):
    return {
        p["name"]: p.get("value") if p.get("value") is not None else p.get("default")
        for p in (datos.get("job_parameters") or [])
    }


def error_job(datos):
    """Devuelve información disponible; no inventa tipo ni capacidad de reintento."""
    if estado_job(datos) not in {"FAILED", "TIMEDOUT", "CANCELED", "INTERNAL_ERROR"}:
        return None

    tareas = datos.get("tasks") or []
    fallidas = [
        t for t in tareas
        if estado_job(t) in {"FAILED", "TIMEDOUT", "CANCELED", "INTERNAL_ERROR"}
    ]
    tarea = next((t for t in fallidas if estado_job(t) == "FAILED"), None)
    if tarea is None:
        tarea = fallidas[0] if fallidas else None

    estado = (tarea or datos).get("state") or {}
    mensaje = estado.get("state_message") or (datos.get("state") or {}).get("state_message")
    detalle = None

    # Recuperar detalles adicionales solo para la Task que falló, si están disponibles.
    if tarea and tarea.get("run_id"):
        try:
            salida = w.jobs.get_run_output(run_id=int(tarea["run_id"])).as_dict()
            mensaje = salida.get("error") or mensaje
            detalle = salida.get("error_trace")
        except Exception:
            pass  # Permisos y límites de la API pueden impedir recuperar el traceback.

    tipo = None
    if mensaje:
        coinc = re.search(r"\b([A-Za-z_][A-Za-z_0-9]*(?:Error|Exception))\b", mensaje)
        tipo = coinc.group(1) if coinc else None

    return {
        "task": tarea.get("task_key") if tarea else None,
        "tipo_error": tipo,
        "mensaje": (mensaje or "La ejecución no terminó correctamente")[:1500],
        "fecha_hora": fecha_local((tarea or datos).get("end_time")),
        "detalle_tecnico": detalle[:3000] if detalle else None,
        "reintentable": None,  # Requiere una política explícita de reintentos.
    }


# 1. JobPadre01 y parámetros efectivos
padre = w.jobs.get_run(run_id=int(RUN_PADRE)).as_dict()
parametros = parametros_job(padre)
parametros_entrada = {
    "max_iteraciones": int(parametros["max_iteraciones"]),
    "tamano_lote": int(parametros["tamano_lote"]),
    "reiniciar_modelos": str(parametros["reiniciar_modelos"]).lower() == "true",
}

# 2. Ubicar la Task For each y las ejecuciones individuales
config_job = w.jobs.get(job_id=int(padre["job_id"])).as_dict()
config_bucle = next(
    t for t in config_job["settings"]["tasks"]
    if t["task_key"] == "bucle_iteraciones"
)
job_hijo_id = int(config_bucle["for_each_task"]["task"]["run_job_task"]["job_id"])
tarea_bucle = next(
    (t for t in padre.get("tasks", []) if t.get("task_key") == "bucle_iteraciones"),
    None,
)
if tarea_bucle and tarea_bucle.get("run_id"):
    bucle = w.jobs.get_run(run_id=int(tarea_bucle["run_id"])).as_dict()
    ids_tasks = {str(t["run_id"]) for t in bucle.get("iterations", [])}
else:
    ids_tasks = set()

# 3. Encontrar los Jobs hijos de ESTA corrida, utilizando trigger_info.run_id
JOBS_HIJOS = {}
if ids_tasks:
    rango = {"job_id": job_hijo_id, "start_time_from": padre["start_time"]}
    if padre.get("end_time"):
        rango["start_time_to"] = padre["end_time"]

    for r in w.jobs.list_runs(**rango):
        d = r.as_dict()
        task_id = str((d.get("trigger_info") or {}).get("run_id"))
        if task_id not in ids_tasks:
            continue
        p = parametros_job(d)
        if p.get("iteracion") is not None:
            JOBS_HIJOS[int(p["iteracion"])] = str(d["run_id"])

if estado_job(padre) == "SUCCEEDED" and len(JOBS_HIJOS) != len(ids_tasks):
    raise RuntimeError(
        "No se encontraron todas las ejecuciones hijas. "
        f"For each: {len(ids_tasks)}, Jobs hijos: {len(JOBS_HIJOS)}"
    )

print("Parámetros:", parametros_entrada)
print("Jobs hijos identificados:", dict(sorted(JOBS_HIJOS.items())))


# COMMAND ----------

def obtener_fila(tabla, job_id, iteracion, seleccionado=False):
    df = spark.table(f"{BASE}.{tabla}").filter(
        (F.col("job_run_id") == str(job_id)) &
        (F.col("iteracion") == int(iteracion))
    )
    if seleccionado:
        df = df.filter(F.col("seleccionado") == True)
    fila = df.first()
    return fila.asDict() if fila is not None else None


def metricas(datos, sufijo=None):
    if datos is None:
        return None
    if sufijo:
        return {
            "auc_roc": datos.get(f"auc_{sufijo}"),
            "f1": datos.get(f"f1_{sufijo}"),
            "accuracy": datos.get(f"accuracy_{sufijo}"),
            "precision": datos.get(f"precision_{sufijo}"),
            "recall": datos.get(f"recall_{sufijo}"),
            "auc_pr": datos.get(f"auc_pr_{sufijo}"),
        }
    return {k: datos.get(src) for k, src in {
        "auc_roc": "auc", "f1": "f1", "accuracy": "accuracy",
        "precision": "precision", "recall": "recall", "auc_pr": "auc_pr",
    }.items()}


def version(valor):
    return int(valor) if valor is not None else None


def json_columna(valor):
    return json.loads(valor) if valor else {}


iteraciones = []
for numero, job_id in sorted(JOBS_HIJOS.items()):
    hijo = w.jobs.get_run(run_id=int(job_id)).as_dict()
    ent = obtener_fila("tabla_resultados_entrenamiento", job_id, numero, seleccionado=True)
    pro = obtener_fila("tabla_promocion", job_id, numero)
    mon = obtener_fila("tabla_monitoreo", job_id, numero) if numero > 1 else None

    challenger = None
    if pro or ent:
        challenger = {
            "algoritmo": (pro or {}).get("algoritmo_challenger") or (ent or {}).get("modelo"),
            "version": version((pro or {}).get("version_challenger") or (ent or {}).get("version_registrada")),
            "metricas": metricas(pro, "challenger") if pro else metricas(ent),
        }

    champion_anterior = None
    if pro and pro.get("version_champion_anterior") is not None:
        champion_anterior = {
            "algoritmo": pro.get("algoritmo_champion_anterior"),
            "version": version(pro.get("version_champion_anterior")),
            "metricas": metricas(pro, "champion_anterior"),
        }

    decision = None
    if pro:
        decision = {
            "ganador": pro.get("ganador"),
            "promovido": pro.get("promovido"),
            "criterio_principal": pro.get("criterio_principal"),
            "criterio_desempate": pro.get("criterio_desempate"),
            "tolerancia_auc": pro.get("tolerancia_auc"),
            "diferencia_auc": pro.get("diferencia_auc"),
            "diferencia_f1": pro.get("diferencia_f1"),
            "motivo": pro.get("motivo"),
            "champion_final": {
                "algoritmo": pro.get("algoritmo_champion_resultante"),
                "version": version(pro.get("version_champion_resultante")),
            },
        }

    drift = {
        "estado": "NO_APLICA" if numero == 1 else "NO_EVALUADO",
        "ks_statistic": None,
        "ks_p_value": None,
        "psi_puntaje": None,
        "psi_variables": None,
    }
    if mon:
        drift = {
            "estado": "SI" if mon.get("hay_drift") else "NO",
            "ks_statistic": mon.get("ks_estadistico"),
            "ks_p_value": mon.get("ks_pvalue"),
            "psi_puntaje": mon.get("psi_puntaje"),
            "psi_variables": json_columna(mon.get("psi_variables_json")),
        }

    item = {
        "iteracion": numero,
        "job_run_id": str(job_id),
        **datos_tiempo(hijo),
        "registros_procesados": ent.get("registros_procesados") if ent else None,
        "challenger": challenger,
        "champion_anterior": champion_anterior,
        "decision": decision,
        "drift": drift,
        "error": error_job(hijo),
    }
    iteraciones.append(item)

fallidas = [i for i in iteraciones if i["estado"] in {"FAILED", "TIMEDOUT", "CANCELED", "INTERNAL_ERROR"}]
error_general = fallidas[0]["error"] if fallidas else error_job(padre)

resultado = {
    "run_id": RUN_PADRE,
    "parametros_entrada": parametros_entrada,
    **datos_tiempo(padre),
    "total_iteraciones": parametros_entrada["max_iteraciones"],
    "iteraciones_completadas": sum(i["estado"] == "SUCCEEDED" for i in iteraciones),
    "iteracion_fallida": fallidas[0]["iteracion"] if fallidas else None,
    "error": error_general,
    "iteraciones": iteraciones,
}


# COMMAND ----------

# JSON consolidado: listo para utilizarse como cuerpo de respuesta de la futura API.
print(json.dumps(resultado, indent=2, ensure_ascii=False, default=str))


# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.apps import AppDeployment

w = WorkspaceClient()

ruta = "/Workspace/MLOPS2026/TallerGeneral/mlops-desaparecidos-api-01"

# Verificar archivos
archivos = [
    x.path.split("/")[-1]
    for x in w.workspace.list(ruta)
]

print("Archivos encontrados:", archivos)

assert {"app.py", "app.yaml", "requirements.txt"}.issubset(set(archivos))

# Iniciar despliegue
w.apps.deploy(
    app_name="mlops-desaparecidos-api-01",
    app_deployment=AppDeployment(source_code_path=ruta)
)

print("Solicitud de despliegue enviada.")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.apps import AppDeployment

w = WorkspaceClient()

w.apps.deploy(
    app_name="mlops-desaparecidos-api-01",
    app_deployment=AppDeployment(
        source_code_path="/Workspace/MLOPS2026/TallerGeneral/mlops-desaparecidos-api-01"
    )
)

print("Despliegue solicitado")

# COMMAND ----------

import requests
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

app_id = w.apps.get(
    "mlops-desaparecidos-api-01"
).oauth2_app_client_id

token_notebook = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook().getContext().apiToken().get()
)

r = requests.post(
    "https://dbc-b52db97b-eac4.cloud.databricks.com/oidc/v1/token",
    data={
        "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
        "subject_token": token_notebook,
        "subject_token_type": "urn:databricks:params:oauth:token-type:personal-access-token",
        "requested_token_type": "urn:ietf:params:oauth:token-type:access_token",
        "scope": "all-apis",
        "audience": app_id
    },
    timeout=30
)

r.raise_for_status()
print(r.json()["access_token"])