import os
from dotenv import load_dotenv
from openai import OpenAI



load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")


MODEL_NAME = "gpt-5.5"


if not API_KEY:
    raise ValueError(
        "No se encontró OPENAI_API_KEY. Revisá que exista el archivo .env "
        "y que tenga tu API key."
    )

client = OpenAI(api_key=API_KEY)


SYSTEM_PROMPT = """
Sos un asistente de pregunta-respuesta usado como baseline experimental.

Tu tarea:
- Responder la pregunta del usuario de forma clara y directa.
- No usar documentos externos.
- No decir que consultaste archivos, corpus, fuentes o bases de datos.
- No inventar citas ni referencias.
- Si la pregunta depende de un documento específico que no fue provisto, indicá que no tenés información suficiente.
- Si la pregunta es ambigua, pedí la aclaración mínima necesaria.

Este sistema representa el baseline S0: LLM directo sin RAG.
"""


def ask_direct_llm(question: str) -> str:
    """
    Envía una pregunta directamente al LLM, sin RAG, sin herramientas
    y sin contexto documental adicional.

    Parameters
    ----------
    question : str
        Pregunta del dataset.

    Returns
    -------
    str
        Respuesta generada por el modelo.
    """

    if not isinstance(question, str) or not question.strip():
        return "ERROR: pregunta vacía o inválida."

    response = client.responses.create(
        model=MODEL_NAME,
        instructions=SYSTEM_PROMPT,
        input=question,
    )

    return response.output_text