"""Inferencia sincrona con el Champion de Unity Catalog.

Valida estrictamente las 12 categorias contra el OneHotEncoder ENTRENADO de la
version Champion vigente. No transforma tildes, mayusculas ni equivalencias y
NO ejecuta Jobs ni reentrena modelos.
"""

from difflib import get_close_matches
from functools import lru_cache
from typing import Any, Mapping
import unicodedata

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

MODELO = "workspace.mlops_desaparecidos.modelo_desaparecidos"
VARIABLE_OBJETIVO = "localizada"
VARIABLES = [
    "Ciudad", "Localidad", "Sexo", "RangoEdad",
    "Escolaridad", "EstadoCivil", "Etnia", "PaisNacimiento",
    "PertenenciaGrupal", "RegistradoPor", "Mes", "Dia",
]


class ErrorValidacionCategorias(ValueError):
    """Categorias desconocidas para la version concreta del Champion."""

    def __init__(self, detalles: list[dict[str, Any]]):
        self.detalles = detalles
        super().__init__("Una o mas categorias no son validas para el Champion vigente.")


def preparar_entrada(datos: Mapping[str, Any]) -> pd.DataFrame:
    """Reproduce la preparacion X = fillna(...).astype(str) de NB04."""
    faltantes = [nombre for nombre in VARIABLES if nombre not in datos]
    desconocidas = [nombre for nombre in datos if nombre not in VARIABLES]
    if faltantes or desconocidas:
        raise ValueError(
            f"Variables faltantes: {faltantes}; variables desconocidas: {desconocidas}"
        )
    # NO normalizar: los modelos actuales se entrenaron con texto exacto.
    return (pd.DataFrame([datos], columns=VARIABLES)
            .fillna("Sin información")
            .astype(str))


@lru_cache(maxsize=3)
def _cargar_modelo(version: str):
    """Conserva en memoria un modelo por version y evita descargarlo en cada POST."""
    return mlflow.sklearn.load_model(f"models:/{MODELO}/{version}")


def _clave_similar(texto: str) -> str:
    """Solo para sugerencias; NUNCA se envia este valor transformado al modelo."""
    sin_tildes = "".join(
        caracter for caracter in unicodedata.normalize("NFKD", texto)
        if not unicodedata.combining(caracter)
    )
    return sin_tildes.strip().casefold()


def _sugerencia(recibido: str, permitidos: list[str]) -> str | None:
    normalizados = {_clave_similar(p): p for p in permitidos}
    buscado = _clave_similar(recibido)
    if buscado in normalizados:
        return normalizados[buscado]
    cercanos = get_close_matches(buscado, list(normalizados), n=1, cutoff=0.65)
    return normalizados[cercanos[0]] if cercanos else None


def validar_categorias(pipeline: Any, X: pd.DataFrame) -> None:
    """Comprueba cada valor frente a las categorias ajustadas del Champion.

    Se ejecuta sobre el pipeline ya cargado, no sobre tablas Delta ni sobre
    listas fijas. Por eso funciona tambien si NB05 promueve otra version.
    """
    try:
        preprocesador = pipeline.named_steps["preprocesamiento"]
        encoder = preprocesador.named_transformers_["categoricas"]
        columnas_encoder = next(
            columnas for nombre, _, columnas in preprocesador.transformers_
            if nombre == "categoricas"
        )
        categorias_ajustadas = encoder.categories_
    except (AttributeError, KeyError, StopIteration, TypeError) as exc:
        raise RuntimeError(
            "El Champion no expone el OneHotEncoder esperado; revisar su pipeline."
        ) from exc

    if len(columnas_encoder) != len(categorias_ajustadas):
        raise RuntimeError("La estructura de categorias del Champion es inconsistente.")

    if set(columnas_encoder) != set(VARIABLES):
        raise RuntimeError(
            "Las variables del Champion no coinciden con las 12 variables de la API."
        )

    errores = []
    for variable, categorias in zip(columnas_encoder, categorias_ajustadas):
        permitido = [str(categoria) for categoria in categorias]
        recibido = str(X.iloc[0][variable])
        if recibido not in permitido:
            errores.append({
                "campo": variable,
                "valor_recibido": recibido,
                "mensaje": "Categoria no reconocida por el Champion vigente.",
                "sugerencia": _sugerencia(recibido, permitido),
                "valores_permitidos": permitido[:25],
                "total_valores_permitidos": len(permitido),
            })

    if errores:
        raise ErrorValidacionCategorias(errores)


def predecir(datos: Mapping[str, Any]) -> dict[str, Any]:
    """Valida contra el Champion actual e infiere una observacion individual."""
    X = preparar_entrada(datos)
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    # Consultar el alias en cada solicitud detecta nuevas promociones.
    champion = MlflowClient().get_model_version_by_alias(MODELO, "champion")
    version = str(champion.version)
    pipeline = _cargar_modelo(version)

    # Antes de inferir, rechazar datos con categorias no vistas durante el fit.
    validar_categorias(pipeline, X)

    clase = int(pipeline.predict(X)[0])
    clases = list(pipeline.classes_)
    if 1 not in clases:
        raise RuntimeError("El Champion no contiene la clase positiva localizada=1")
    indice_positivo = clases.index(1)
    probabilidad = float(pipeline.predict_proba(X)[0, indice_positivo])

    return {
        "variable_objetivo": VARIABLE_OBJETIVO,
        "prediccion": clase,
        "probabilidad": probabilidad,  # P(localizada=1), no Accuracy/F1
        "modelo": champion.tags.get("algoritmo", type(pipeline.steps[-1][1]).__name__),
        "version_champion": int(version),
    }
