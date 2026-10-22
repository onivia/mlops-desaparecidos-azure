"""API REST del proyecto MLOps Desaparecidos.

POST /api/entrenar dispara JobPadre01; GET /api/entrenar/{run_id}
consulta Jobs y tablas Delta. No ejecuta notebooks dentro de esta app.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from time import perf_counter
from time import monotonic, sleep
from zoneinfo import ZoneInfo

from databricks.sdk import WorkspaceClient
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from pydantic import BaseModel, ConfigDict, Field
from fastapi.responses import JSONResponse
from inferencia import ErrorValidacionCategorias, predecir

logger = logging.getLogger("mlops-api")
app = FastAPI(title="API MLOps - Desaparecidos", version="1.0.0")
ZONA = ZoneInfo("America/Bogota")


def cliente():
    # Databricks Apps suministra las credenciales de su service principal.
    return WorkspaceClient()


def recurso(nombre):
    valor = os.getenv(nombre)
    if not valor:
        raise RuntimeError(f"Falta configurar el recurso {nombre} en app.yaml")
    return valor


class EntradaEjecucion(BaseModel):
    max_iteraciones: int = Field(ge=1, le=50)
    tamano_lote: int = Field(ge=1, le=100000)
    reiniciar_modelos: bool


class EntradaPrediccion(BaseModel):
    """Una fila de las mismas doce variables empleadas en NB04."""
    model_config = ConfigDict(extra="forbid")

    Ciudad: str = Field(min_length=1)
    Localidad: str = Field(min_length=1)
    Sexo: str = Field(min_length=1)
    RangoEdad: str = Field(min_length=1)
    Escolaridad: str = Field(min_length=1)
    EstadoCivil: str = Field(min_length=1)
    Etnia: str = Field(min_length=1)
    PaisNacimiento: str = Field(min_length=1)
    PertenenciaGrupal: str = Field(min_length=1)
    RegistradoPor: str = Field(min_length=1)
    Mes: str = Field(min_length=1)
    Dia: str = Field(min_length=1)


@app.middleware("http")
async def medir_solicitud_prediccion(request: Request, call_next):
    """Marca el inicio antes de validar el JSON, solo para POST /api/predecir."""
    if request.method == "POST" and request.url.path == "/api/predecir":
        request.state.inicio_prediccion = datetime.now(ZONA)
        request.state.t0_prediccion = perf_counter()
    return await call_next(request)


def tiempos_prediccion(inicio: datetime, t0: float) -> dict:
    fin = datetime.now(ZONA)
    return {
        "fecha_inicio": inicio.isoformat(timespec="seconds"),
        "fecha_fin": fin.isoformat(timespec="seconds"),
        "duracion": f"{perf_counter() - t0:.2f}s",
    }


def respuesta_error_prediccion(
    *, codigo_http: int, inicio: datetime, t0: float,
    tipo: str, mensaje: str, detalles: list | None = None,
) -> JSONResponse:
    error = {"tipo": tipo, "mensaje": mensaje}
    if detalles is not None:
        error["detalles"] = detalles
    return JSONResponse(
        status_code=codigo_http,
        content={
            "estado": "FAILED",
            "variable_objetivo": "localizada",
            "prediccion": None,
            "probabilidad": None,
            "modelo": None,
            "version_champion": None,
            **tiempos_prediccion(inicio, t0),
            "error": error,
        },
    )


@app.exception_handler(RequestValidationError)
async def validar_json_prediccion(request: Request, exc: RequestValidationError):
    """Errores Pydantic del POST de prediccion usan el mismo contrato JSON.

    No cambia el formato nativo de validacion de los otros endpoints.
    """
    if request.method != "POST" or request.url.path != "/api/predecir":
        return await request_validation_exception_handler(request, exc)

    detalles = []
    for error in exc.errors():
        ruta = [str(elemento) for elemento in error.get("loc", ()) if elemento != "body"]
        campo = ".".join(ruta) or "body"
        tipo = error.get("type", "invalid")
        valor = error.get("input")
        if tipo == "missing":
            mensaje = "Campo obligatorio ausente."
            valor = None  # Pydantic devuelve el body completo: no exponerlo
        elif tipo == "extra_forbidden":
            mensaje = "Campo no permitido."
        elif tipo == "json_invalid":
            mensaje = "El cuerpo de la solicitud no contiene un JSON valido."
            valor = None
        else:
            mensaje = "Valor o tipo de dato no valido para este campo."
        # Evitar reflejar payloads completos o datos anidados en el error.
        if not isinstance(valor, (str, int, float, bool)):
            valor = None
        elif isinstance(valor, str):
            valor = valor[:150]
        detalles.append({
            "campo": campo,
            "valor_recibido": valor,
            "mensaje": mensaje,
            "tipo_validacion": tipo,
        })
    inicio = getattr(request.state, "inicio_prediccion", datetime.now(ZONA))
    t0 = getattr(request.state, "t0_prediccion", perf_counter())
    return respuesta_error_prediccion(
        codigo_http=422,
        inicio=inicio,
        t0=t0,
        tipo="DATOS_ENTRADA_INVALIDOS",
        mensaje="La solicitud contiene variables faltantes o con formato invalido.",
        detalles=detalles,
    )


@app.post("/api/predecir")
def predecir_caso(entrada: EntradaPrediccion, request: Request):
    """Prediccion sincronica: NO inicia Jobs ni modifica el Champion."""
    inicio = request.state.inicio_prediccion
    t0 = request.state.t0_prediccion
    try:
        resultado = predecir(entrada.model_dump())
        return {
            "estado": "SUCCEEDED",
            **resultado,
            **tiempos_prediccion(inicio, t0),
            "error": None,
        }
    except ErrorValidacionCategorias as exc:
        logger.info("Prediccion rechazada: categorias no reconocidas (%s campos)", len(exc.detalles))
        return respuesta_error_prediccion(
            codigo_http=422,
            inicio=inicio,
            t0=t0,
            tipo="CATEGORIA_INVALIDA",
            mensaje=str(exc),
            detalles=exc.detalles,
        )
    except Exception as exc:
        logger.exception("Fallo en POST /api/predecir")
        return respuesta_error_prediccion(
            codigo_http=503,
            inicio=inicio,
            t0=t0,
            tipo=type(exc).__name__,
            mensaje="No fue posible ejecutar la prediccion. Consulte los logs de Databricks Apps.",
        )



@app.get("/health", include_in_schema=False)
@app.get("/api/health")
def health():
    return {"estado": "OK", "servicio": "mlops-desaparecidos-api-01"}


@app.post("/entrenar", status_code=status.HTTP_202_ACCEPTED, include_in_schema=False)
@app.post("/api/entrenar", status_code=status.HTTP_202_ACCEPTED)
def iniciar_ejecucion(entrada: EntradaEjecucion):
    try:
        respuesta = cliente().jobs.run_now(
            job_id=int(recurso("JOB_PADRE_ID")),
            job_parameters={
                "max_iteraciones": str(entrada.max_iteraciones),
                "tamano_lote": str(entrada.tamano_lote),
                "reiniciar_modelos": str(entrada.reiniciar_modelos).lower(),
            },
        )
        run_id = respuesta.response.run_id
        if run_id is None:
            raise RuntimeError("Jobs no devolvio run_id")
    except Exception as exc:
        logger.exception("Error al iniciar JobPadre01")
        raise HTTPException(502, "No fue posible iniciar JobPadre01. Consulte los logs de la App.") from exc
    return {
        "run_id": str(run_id),
        "estado": "PENDING",
        "parametros_entrada": entrada.model_dump(),
        "consulta": f"/api/entrenar/{run_id}",
    }


def estado_job(datos):
    state = datos.get("state") or {}
    resultado = state.get("result_state")
    ciclo = state.get("life_cycle_state")
    return {
        "SUCCESS": "SUCCEEDED",
        "SUCCESS_WITH_FAILURES": "SUCCEEDED_WITH_FAILURES",
        "UPSTREAM_FAILED": "FAILED",
        "MAXIMUM_CONCURRENT_RUNS_REACHED": "FAILED",
    }.get(resultado, resultado or ciclo or "UNKNOWN")


def fecha_local(ms):
    if not ms:
        return None
    return datetime.fromtimestamp(ms / 1000, timezone.utc).astimezone(ZONA).isoformat(timespec="seconds")


def datos_tiempo(datos):
    inicio, fin = datos.get("start_time"), datos.get("end_time")
    segundos = max(0, (fin - inicio) // 1000) if inicio is not None and fin is not None else None
    return {
        "estado": estado_job(datos),
        "fecha_inicio": fecha_local(inicio),
        "fecha_fin": fecha_local(fin),
        "duracion": f"{segundos // 60}m{segundos % 60}s" if segundos is not None else None,
    }


def parametros_job(datos):
    return {
        p["name"]: p.get("value") if p.get("value") is not None else p.get("default")
        for p in (datos.get("job_parameters") or [])
    }


def error_job(w, datos):
    if estado_job(datos) not in {"FAILED", "TIMEDOUT", "CANCELED", "INTERNAL_ERROR", "SUCCEEDED_WITH_FAILURES"}:
        return None
    tareas = datos.get("tasks") or []
    fallidas = [
        t for t in tareas
        if estado_job(t) in {"FAILED", "TIMEDOUT", "CANCELED", "INTERNAL_ERROR"}
    ]
    tarea = next((t for t in fallidas if estado_job(t) == "FAILED"), None)
    tarea = tarea or (fallidas[0] if fallidas else None)
    estado = (tarea or datos).get("state") or {}
    mensaje = estado.get("state_message") or (datos.get("state") or {}).get("state_message")
    detalle = None
    if tarea and tarea.get("run_id"):
        try:
            salida = w.jobs.get_run_output(run_id=int(tarea["run_id"])).as_dict()
            mensaje = salida.get("error") or mensaje
            detalle = salida.get("error_trace")
        except Exception:
            logger.warning("No fue posible consultar get_run_output", exc_info=True)
    tipo = None
    if mensaje:
        m = re.search(r"\b([A-Za-z_][A-Za-z_0-9]*(?:Error|Exception))\b", mensaje)
        tipo = m.group(1) if m else None
    return {
        "task": tarea.get("task_key") if tarea else None,
        "tipo_error": tipo,
        "mensaje": (mensaje or "La ejecución no terminó correctamente")[:1500],
        "fecha_hora": fecha_local((tarea or datos).get("end_time")),
        "detalle_tecnico": detalle[:3000] if detalle else None,
        "reintentable": None,  # No inferir la posibilidad de reintento.
    }


def leer_fila_sql(w, tabla, job_run_id, iteracion, seleccionado=False):
    """Consulta una fila Delta por SQL Warehouse (Databricks Apps no tiene Spark)."""
    if not str(job_run_id).isdigit() or not isinstance(iteracion, int):
        raise ValueError("Identificadores inválidos para consultar Delta")
    # Nombre proviene solo de app.yaml, configurado mediante recursos registrados.
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*", tabla):
        raise ValueError("Nombre de tabla incorrecto")
    where = f"job_run_id = '{job_run_id}' AND iteracion = {iteracion}"
    if seleccionado:
        where += " AND seleccionado = TRUE"
    sql = f"SELECT * FROM {tabla} WHERE {where} LIMIT 2"
    r = w.statement_execution.execute_statement(
        warehouse_id=recurso("WAREHOUSE_ID"),
        statement=sql,
        wait_timeout="30s",
        row_limit=2,
    ).as_dict()
    limite = monotonic() + 60
    while (r.get("status") or {}).get("state") in {"PENDING", "RUNNING"}:
        if monotonic() >= limite:
            raise TimeoutError(f"La consulta SQL tardó demasiado: {tabla}")
        sleep(2)
        r = w.statement_execution.get_statement(statement_id=r["statement_id"]).as_dict()
    estado = (r.get("status") or {}).get("state")
    if estado != "SUCCEEDED":
        raise RuntimeError(f"SQL no finalizó correctamente ({tabla}): {r.get('status')}")
    filas = ((r.get("result") or {}).get("data_array") or [])
    if not filas:
        return None
    columnas = ((r.get("manifest") or {}).get("schema") or {}).get("columns") or []
    salida = {}
    for col, valor in zip(columnas, filas[0]):
        nombre = col["name"]
        tipo = str(col.get("type_name") or "").upper()
        if valor is None:
            salida[nombre] = None
        elif tipo in {"BOOLEAN"}:
            salida[nombre] = str(valor).lower() == "true"
        elif tipo in {"LONG", "BIGINT", "INT", "INTEGER", "SHORT", "BYTE"}:
            salida[nombre] = int(valor)
        elif tipo in {"DOUBLE", "FLOAT", "DECIMAL"}:
            salida[nombre] = float(valor)
        else:
            salida[nombre] = valor
    return salida


def metricas(datos, sufijo=None):
    if not datos:
        return None
    campos = {
        "auc_roc": "auc", "f1": "f1", "accuracy": "accuracy",
        "precision": "precision", "recall": "recall", "auc_pr": "auc_pr",
    }
    return {k: datos.get(f"{v}_{sufijo}" if sufijo else v) for k, v in campos.items()}


def version(valor):
    return int(valor) if valor is not None else None


def obtener_hijos(w, padre):
    config = w.jobs.get(job_id=int(padre["job_id"])).as_dict()
    bucle_config = next(
        t for t in config["settings"]["tasks"] if t["task_key"] == "bucle_iteraciones"
    )
    job_hijo_id = int(bucle_config["for_each_task"]["task"]["run_job_task"]["job_id"])
    # Comprobar que el ID corresponde al Job hijo autorizado en nuestra App.
    if job_hijo_id != int(recurso("JOB_HIJO_ID")):
        raise RuntimeError("El Job Hijo configurado no coincide con el del pipeline")
    task = next(
        (t for t in padre.get("tasks", []) if t.get("task_key") == "bucle_iteraciones"), None
    )
    if not task or not task.get("run_id"):
        return {}
    bucle = w.jobs.get_run(run_id=int(task["run_id"])).as_dict()
    ids_tasks = {str(t["run_id"]) for t in bucle.get("iterations", [])}
    if not ids_tasks:
        return {}
    rango = {"job_id": job_hijo_id, "start_time_from": padre["start_time"]}
    if padre.get("end_time"):
        rango["start_time_to"] = padre["end_time"]
    hijos = {}
    for run in w.jobs.list_runs(**rango):
        d = run.as_dict()
        task_id = str((d.get("trigger_info") or {}).get("run_id"))
        if task_id not in ids_tasks:
            continue
        par = parametros_job(d)
        if par.get("iteracion") is not None:
            hijos[int(par["iteracion"])] = str(d["run_id"])
    if estado_job(padre) == "SUCCEEDED" and len(hijos) != len(ids_tasks):
        raise RuntimeError("Faltan Jobs hijos en una ejecución ya terminada")
    return hijos


def obtener_resultados(w, run_id):
    padre = w.jobs.get_run(run_id=run_id).as_dict()
    if int(padre.get("job_id") or 0) != int(recurso("JOB_PADRE_ID")):
        raise HTTPException(status_code=404, detail="No existe esa ejecución de JobPadre01")
    par = parametros_job(padre)
    entrada = {
        "max_iteraciones": int(par["max_iteraciones"]),
        "tamano_lote": int(par["tamano_lote"]),
        "reiniciar_modelos": str(par["reiniciar_modelos"]).lower() == "true",
    }
    hijos = obtener_hijos(w, padre)
    iteraciones = []
    for numero, job_id in sorted(hijos.items()):
        hijo = w.jobs.get_run(run_id=int(job_id)).as_dict()
        terminado = estado_job(hijo) in {"SUCCEEDED", "FAILED", "TIMEDOUT", "CANCELED", "INTERNAL_ERROR", "SUCCEEDED_WITH_FAILURES"}
        ent = pro = mon = None
        if terminado:
            ent = leer_fila_sql(w, recurso("TABLA_ENTRENAMIENTO"), job_id, numero, seleccionado=True)
            pro = leer_fila_sql(w, recurso("TABLA_PROMOCION"), job_id, numero)
            if numero > 1:
                mon = leer_fila_sql(w, recurso("TABLA_MONITOREO"), job_id, numero)
        challenger = None
        if pro or ent:
            challenger = {
                "algoritmo": (pro or {}).get("algoritmo_challenger") or (ent or {}).get("modelo"),
                "version": version((pro or {}).get("version_challenger") or (ent or {}).get("version_registrada")),
                "metricas": metricas(pro, "challenger") if pro else metricas(ent),
            }
        anterior = None
        if pro and pro.get("version_champion_anterior") is not None:
            anterior = {
                "algoritmo": pro.get("algoritmo_champion_anterior"),
                "version": version(pro.get("version_champion_anterior")),
                "metricas": metricas(pro, "champion_anterior"),
            }
        decision = None
        if pro:
            decision = {
                "ganador": pro.get("ganador"), "promovido": pro.get("promovido"),
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
            "ks_statistic": None, "ks_p_value": None,
            "psi_puntaje": None, "psi_variables": None,
        }
        if mon:
            drift = {
                "estado": "SI" if mon.get("hay_drift") else "NO",
                "ks_statistic": mon.get("ks_estadistico"),
                "ks_p_value": mon.get("ks_pvalue"),
                "psi_puntaje": mon.get("psi_puntaje"),
                "psi_variables": json.loads(mon.get("psi_variables_json") or "{}"),
            }
        iteraciones.append({
            "iteracion": numero, "job_run_id": job_id,
            **datos_tiempo(hijo),
            "registros_procesados": ent.get("registros_procesados") if ent else None,
            "challenger": challenger, "champion_anterior": anterior,
            "decision": decision, "drift": drift, "error": error_job(w, hijo),
        })
    fallidas = [
        i for i in iteraciones
        if i["estado"] in {"FAILED", "TIMEDOUT", "CANCELED", "INTERNAL_ERROR"}
    ]
    return {
        "run_id": str(run_id),
        "parametros_entrada": entrada,
        **datos_tiempo(padre),
        "total_iteraciones": entrada["max_iteraciones"],
        "iteraciones_completadas": sum(i["estado"] == "SUCCEEDED" for i in iteraciones),
        "iteracion_fallida": fallidas[0]["iteracion"] if fallidas else None,
        "error": fallidas[0]["error"] if fallidas else error_job(w, padre),
        "iteraciones": iteraciones,
    }


@app.get("/entrenar/{run_id}", include_in_schema=False)
@app.get("/api/entrenar/{run_id}")
def consultar_ejecucion(run_id: int):
    if run_id <= 0:
        raise HTTPException(422, "run_id debe ser positivo")
    try:
        return obtener_resultados(cliente(), run_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error al consultar los resultados de %s", run_id)
        raise HTTPException(
            502, "No fue posible consultar la ejecución. Revise permisos, recursos y logs de la App."
        ) from exc

# ---------- PRUEBA TEMPORAL: acceso al Champion desde Databricks Apps ----------
# Quitar este endpoint una vez terminada la validacion.
@app.get("/api/diagnostico/champion", include_in_schema=False)
def diagnostico_champion():
    """Verifica acceso al alias y carga del artefacto MLflow sin entrenar ni predecir."""
    from time import perf_counter

    inicio = perf_counter()
    fase = "importar_mlflow"
    try:
        import mlflow
        import mlflow.sklearn
        from mlflow.tracking import MlflowClient

        fase = "configurar_mlflow"
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
        nombre = "workspace.mlops_desaparecidos.modelo_desaparecidos"

        fase = "resolver_alias_champion"
        version = MlflowClient().get_model_version_by_alias(nombre, "champion")

        fase = "cargar_artefacto_mlflow"
        modelo = mlflow.sklearn.load_model(f"models:/{nombre}/{version.version}")

        return {
            "estado": "SUCCEEDED",
            "modelo_registrado": nombre,
            "alias": "champion",
            "version_champion": int(version.version),
            "carga_modelo": "OK",
            "tipo_objeto": type(modelo).__name__,
            "pasos_pipeline": list(getattr(modelo, "named_steps", {}).keys()),
            "soporta_predict": callable(getattr(modelo, "predict", None)),
            "soporta_predict_proba": callable(getattr(modelo, "predict_proba", None)),
            "duracion_segundos": round(perf_counter() - inicio, 2),
        }
    except Exception as exc:
        logger.exception("Diagnostico Champion fallo en la fase: %s", fase)
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=503, content={
            "estado": "FAILED",
            "fase": fase,
            "tipo_error": type(exc).__name__,
            "error": "Revisar los logs de Databricks Apps para ver el detalle tecnico."
        })
# ---------- FIN DE PRUEBA TEMPORAL ----------
