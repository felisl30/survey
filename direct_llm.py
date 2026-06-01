"""
direct_llm.py

Cliente del baseline S0: LLM directo, sin RAG, sin herramientas y sin
contexto documental externo.

Este módulo expone dos funciones:
    - ask_direct_llm(prompt): mantiene compatibilidad con el runner viejo.
    - ask_direct_llm_with_metadata(prompt): devuelve respuesta + usage + metadata.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from dotenv import load_dotenv
load_dotenv()

MODEL_NAME = "gpt-5.5"


SYSTEM_PROMPT = """
Sos un asistente de pregunta-respuesta usado como baseline experimental.

Tu tarea:
- Responder la pregunta del usuario de forma clara y directa.
- No usar documentos externos.
- No decir que consultaste archivos, corpus, fuentes o bases de datos.
- No inventar citas ni referencias.
- Respetar estrictamente el formato de salida solicitado por el prompt del usuario.
- Si el prompt pide JSON, devolvé únicamente JSON válido.
- Si la pregunta depende de un documento específico que no fue provisto, indicá que no tenés información suficiente.
- Si la pregunta es ambigua, pedí la aclaración mínima necesaria.

Este sistema representa el baseline S0: LLM directo sin RAG.
""".strip()


def _get_client():
    """
    Crea el cliente de OpenAI de forma lazy.

    Ventaja:
        Permite importar este módulo aunque todavía no exista OPENAI_API_KEY.
        El error aparece recién cuando se intenta llamar al modelo.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "No se encontró OPENAI_API_KEY. Revisá que exista el archivo .env "
            "o que la variable esté exportada en el entorno."
        )

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "No se encontró el paquete 'openai'. Instalalo con: pip install openai"
        ) from exc

    return OpenAI(api_key=api_key)


def _jsonable(value: Any) -> Any:
    """
    Convierte objetos del SDK a estructuras serializables en JSON.
    """
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]

    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())

    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())

    return str(value)


def _extract_usage_fields(usage: Any) -> dict[str, int | None]:
    """
    Extrae usage de forma robusta, porque puede venir como objeto o dict.
    """
    usage_json = _jsonable(usage)

    if not isinstance(usage_json, dict):
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        }

    input_tokens = usage_json.get("input_tokens")
    output_tokens = usage_json.get("output_tokens")
    total_tokens = usage_json.get("total_tokens")

    # Fallback por si una versión del SDK usa nombres alternativos.
    if input_tokens is None:
        input_tokens = usage_json.get("prompt_tokens")
    if output_tokens is None:
        output_tokens = usage_json.get("completion_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def ask_direct_llm_with_metadata(
    prompt: str,
    *,
    model: str | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    max_retries: int = 2,
    retry_base_seconds: float = 2.0,
) -> dict[str, Any]:
    """
    Envía un prompt directamente al LLM, sin RAG ni herramientas.

    Parameters
    ----------
    prompt:
        Prompt final que se enviará al modelo.
    model:
        Nombre del modelo. Si no se pasa, usa OPENAI_MODEL o MODEL_NAME.
    system_prompt:
        Instrucciones generales del baseline S0.
    max_retries:
        Cantidad máxima de reintentos ante errores transitorios.
    retry_base_seconds:
        Espera base para backoff exponencial.

    Returns
    -------
    dict
        Diccionario con respuesta cruda, usage y metadatos.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt vacío o inválido.")

    selected_model = model or MODEL_NAME
    client = _get_client()

    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        start_time = time.time()

        try:
            response = client.responses.create(
                model=selected_model,
                instructions=system_prompt,
                input=prompt,
            )

            latency_seconds = round(time.time() - start_time, 3)
            raw_output = getattr(response, "output_text", "")

            usage = getattr(response, "usage", None)
            usage_json = _jsonable(usage)
            usage_fields = _extract_usage_fields(usage)

            return {
                "model": selected_model,
                "raw_output": raw_output,
                "latency_seconds": latency_seconds,
                "usage_json": json.dumps(usage_json, ensure_ascii=False),
                "input_tokens": usage_fields["input_tokens"],
                "output_tokens": usage_fields["output_tokens"],
                "total_tokens": usage_fields["total_tokens"],
            }

        except Exception as exc:
            last_error = exc

            if attempt >= max_retries:
                break

            sleep_seconds = retry_base_seconds * (2 ** attempt)
            time.sleep(sleep_seconds)

    raise RuntimeError(f"Falló el llamado al modelo luego de {max_retries + 1} intentos: {last_error}")


def ask_direct_llm(prompt: str) -> str:
    """
    Wrapper compatible con el runner inicial.

    Devuelve únicamente el texto generado por el modelo.
    """
    result = ask_direct_llm_with_metadata(prompt)
    return result["raw_output"]
