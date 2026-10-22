"""FastAPI de Azure: proxy transparente hacia la App MLOps de Databricks.

Conserva métodos, rutas, cuerpos JSON, códigos HTTP y respuestas Databricks.
La autenticación OAuth M2M se ejecuta exclusivamente en el servidor Azure.
"""

import asyncio
import logging
import os
import time

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

logger = logging.getLogger("mlops-azure-proxy")
app = FastAPI(title="API MLOps Desaparecidos - Azure", version="1.0.0")

# Al publicar el HTML, configurar en Azure el origen exacto, sin barra final.
frontend_origin = os.getenv("FRONTEND_ORIGIN", "").strip().rstrip("/")
if frontend_origin:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[frontend_origin],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

WORKSPACE_URL = os.getenv("DATABRICKS_WORKSPACE_URL", "").strip().rstrip("/")
APP_URL = os.getenv("DATABRICKS_APP_URL", "").strip().rstrip("/")
CLIENT_ID = os.getenv("DATABRICKS_CLIENT_ID", "").strip()
CLIENT_SECRET = os.getenv("DATABRICKS_CLIENT_SECRET", "")

_token = None
_token_expiration = 0.0
_token_lock = asyncio.Lock()


def _configurado() -> bool:
    return all((WORKSPACE_URL, APP_URL, CLIENT_ID, CLIENT_SECRET))


async def _obtener_token() -> str:
    global _token, _token_expiration
    if not _configurado():
        raise HTTPException(status_code=503, detail="Conexión Databricks no configurada en Azure")

    async with _token_lock:
        if _token and time.monotonic() < _token_expiration:
            return _token

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                respuesta = await client.post(
                    f"{WORKSPACE_URL}/oidc/v1/token",
                    auth=(CLIENT_ID, CLIENT_SECRET),
                    data={"grant_type": "client_credentials", "scope": "all-apis"},
                )
                respuesta.raise_for_status()
                datos = respuesta.json()
                token = datos["access_token"]
                duracion = int(datos.get("expires_in", 3600))
        except (httpx.HTTPError, KeyError, ValueError, TypeError):
            logger.exception("Error al obtener token OAuth de Databricks")
            raise HTTPException(status_code=502, detail="Fallo de autenticación con Databricks")

        _token, _token_expiration = token, time.monotonic() + max(0, duracion - 60)
        return token


async def _reenviar(request: Request, ruta: str) -> Response:
    token = await _obtener_token()
    cabeceras = {"Authorization": f"Bearer {token}"}
    if "content-type" in request.headers:
        cabeceras["Content-Type"] = request.headers["content-type"]

    try:
        async with httpx.AsyncClient(timeout=180.0, follow_redirects=False) as client:
            respuesta = await client.request(
                request.method,
                f"{APP_URL}{ruta}",
                params=request.query_params,
                content=await request.body(),
                headers=cabeceras,
            )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Tiempo de espera agotado al consultar Databricks")
    except httpx.RequestError:
        logger.exception("Error de comunicación con Databricks")
        raise HTTPException(status_code=502, detail="No fue posible conectar con Databricks")

    # No transforma el JSON ni sustituye sus campos; transmite también los HTTP 4xx/5xx.
    tipo = respuesta.headers.get("content-type", "application/json")
    return Response(content=respuesta.content, status_code=respuesta.status_code,
                    headers={"Content-Type": tipo})


@app.get("/health")
async def health():
    """Diagnóstico local de Azure; no invoca Databricks."""
    return {"estado": "OK" if _configurado() else "NO_CONFIGURADO",
            "servicio": "mlops-desaparecidos-azure-proxy"}


@app.post("/api/entrenar")
async def entrenar(request: Request):
    return await _reenviar(request, "/api/entrenar")


@app.get("/api/entrenar/{run_id}")
async def consultar_entrenamiento(run_id: str, request: Request):
    return await _reenviar(request, f"/api/entrenar/{run_id}")


@app.post("/api/predecir")
async def predecir(request: Request):
    return await _reenviar(request, "/api/predecir")
