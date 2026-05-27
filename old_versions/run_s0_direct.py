import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from old_versions.direct_llm import ask_direct_llm


INPUT_PATH = Path("data/questions.csv")
OUTPUT_PATH = Path("outputs/results_s0_direct.csv")


def run_experiment() -> None:
    """
    Corre el baseline S0 sobre todas las preguntas del dataset.

    Entrada:
        data/questions.csv

    Salida:
        outputs/results_s0_direct.csv
    """

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    required_columns = {"id", "case_type", "question"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Faltan columnas obligatorias en el CSV: {missing_columns}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Running S0 Direct LLM"):
        question_id = row["id"]
        case_type = row["case_type"]
        question = row["question"]
        expected_answer = row.get("expected_answer", "")

        start_time = time.time()

        try:
            answer = ask_direct_llm(question)
            error = ""
        except Exception as exc:
            answer = ""
            error = str(exc)

        end_time = time.time()
        latency_seconds = round(end_time - start_time, 3)

        results.append(
            {
                "id": question_id,
                "system": "S0_direct_llm",
                "case_type": case_type,
                "question": question,
                "expected_answer": expected_answer,
                "answer": answer,
                "latency_seconds": latency_seconds,
                "error": error,
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nResultados guardados en: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_experiment()