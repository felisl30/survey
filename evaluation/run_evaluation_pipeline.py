#!/usr/bin/env python3
"""
run_evaluation_pipeline.py

Ejecuta S1, S2 y S3 sobre el mismo set de preguntas de evaluación unificado
y guarda los resultados evaluados en outputs/eval/.

Para cada sistema corre:
  raw runner → parser → evaluator(es)

El script NO reimplementa lógica de RAG; invoca los runners existentes
vía subprocess con los flags correctos.

Uso:
    python evaluation/run_evaluation_pipeline.py
    python evaluation/run_evaluation_pipeline.py --systems s1,s2 --limit 5   # smoke test
    python evaluation/run_evaluation_pipeline.py --model gpt-4o-mini --force
    python evaluation/run_evaluation_pipeline.py --resume                     # continúa si hay raw ya guardado
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import (
    EVAL_INDEX_DIR,
    EVAL_OUTPUTS_DIR,
    EVAL_QRELS_PATH,
    EVAL_QUESTIONS_PATH,
)

PYTHON = sys.executable

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str], label: str) -> None:
    print(f"\n>>> {label}")
    print("    " + " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Falló con código {result.returncode}: {label}")


def model_tag(model: str | None) -> str:
    if not model:
        return "default"
    return model.replace("/", "_").replace("-", "_").replace(".", "_")


# ---------------------------------------------------------------------------
# Per-system pipelines
# ---------------------------------------------------------------------------

def run_s1(
    questions_path: Path,
    index_dir: Path,
    output_dir: Path,
    model: str | None,
    limit: int | None,
    resume: bool,
    force: bool,
) -> Path:
    tag = model_tag(model)
    raw_path = output_dir / f"s1_{tag}_raw.csv"
    parsed_path = output_dir / f"s1_{tag}_parsed.csv"
    results_path = output_dir / f"s1_{tag}_results.csv"
    summary_path = output_dir / f"s1_{tag}_summary.json"
    group_summary_path = output_dir / f"s1_{tag}_summary_by_group.csv"

    if not raw_path.exists() or force:
        cmd = [
            PYTHON, str(PROJECT_ROOT / "s1_model_code" / "run_s1_rag.py"),
            "--input-path", str(questions_path),
            "--index-dir", str(index_dir),
            "--output-path", str(raw_path),
        ]
        if model:
            cmd += ["--model", model]
        if limit is not None:
            cmd += ["--limit", str(limit)]
        if resume:
            cmd.append("--resume")
        run(cmd, "S1 raw")
    else:
        print(f"[S1] raw ya existe, skip: {raw_path}")

    run([
        PYTHON, str(PROJECT_ROOT / "s1_model_code" / "parse_s1_outputs.py"),
        "--input-path", str(raw_path),
        "--output-path", str(parsed_path),
    ], "S1 parse")

    run([
        PYTHON, str(PROJECT_ROOT / "s1_model_code" / "evaluate_s1_answers.py"),
        "--input-path", str(parsed_path),
        "--output-path", str(results_path),
        "--summary-path", str(summary_path),
        "--group-summary-path", str(group_summary_path),
    ], "S1 evaluate")

    return results_path


def run_s2(
    questions_path: Path,
    index_dir: Path,
    output_dir: Path,
    model: str | None,
    limit: int | None,
    resume: bool,
    force: bool,
) -> Path:
    tag = model_tag(model)
    raw_path = output_dir / f"s2_{tag}_raw.csv"
    parsed_path = output_dir / f"s2_{tag}_parsed.csv"
    results_path = output_dir / f"s2_{tag}_results.csv"
    summary_path = output_dir / f"s2_{tag}_summary.json"
    group_summary_path = output_dir / f"s2_{tag}_summary_by_group.csv"
    routing_results_path = output_dir / f"s2_{tag}_routing_results.csv"
    routing_summary_path = output_dir / f"s2_{tag}_routing_summary.json"
    routing_group_path = output_dir / f"s2_{tag}_routing_summary_by_group.csv"

    if not raw_path.exists() or force:
        cmd = [
            PYTHON, str(PROJECT_ROOT / "s2_model_code" / "run_s2_adaptive_rag.py"),
            "--input-path", str(questions_path),
            "--index-dir", str(index_dir),
            "--output-path", str(raw_path),
        ]
        if model:
            cmd += ["--model", model]
        if limit is not None:
            cmd += ["--limit", str(limit)]
        if resume:
            cmd.append("--resume")
        run(cmd, "S2 raw")
    else:
        print(f"[S2] raw ya existe, skip: {raw_path}")

    run([
        PYTHON, str(PROJECT_ROOT / "s2_model_code" / "parse_s2_outputs.py"),
        "--input-path", str(raw_path),
        "--output-path", str(parsed_path),
    ], "S2 parse")

    run([
        PYTHON, str(PROJECT_ROOT / "s2_model_code" / "evaluate_s2_routing.py"),
        "--input-path", str(parsed_path),
        "--output-path", str(routing_results_path),
        "--summary-path", str(routing_summary_path),
        "--group-summary-path", str(routing_group_path),
    ], "S2 evaluate routing")

    run([
        PYTHON, str(PROJECT_ROOT / "s2_model_code" / "evaluate_s2_answers.py"),
        "--input-path", str(parsed_path),
        "--output-path", str(results_path),
        "--summary-path", str(summary_path),
        "--group-summary-path", str(group_summary_path),
    ], "S2 evaluate answers")

    return results_path


def run_s3(
    questions_path: Path,
    index_dir: Path,
    output_dir: Path,
    model: str | None,
    limit: int | None,
    resume: bool,
    force: bool,
) -> Path:
    tag = model_tag(model)
    raw_path = output_dir / f"s3_{tag}_raw.csv"
    parsed_path = output_dir / f"s3_{tag}_parsed.csv"
    results_path = output_dir / f"s3_{tag}_results.csv"
    summary_path = output_dir / f"s3_{tag}_summary.json"
    group_summary_path = output_dir / f"s3_{tag}_summary_by_group.csv"

    if not raw_path.exists() or force:
        cmd = [
            PYTHON, str(PROJECT_ROOT / "s3_model_code" / "run_s3_flare_like.py"),
            "--input-path", str(questions_path),
            "--index-dir", str(index_dir),
            "--output-path", str(raw_path),
        ]
        if model:
            cmd += ["--model", model]
        if limit is not None:
            cmd += ["--limit", str(limit)]
        if resume:
            cmd.append("--resume")
        run(cmd, "S3 raw")
    else:
        print(f"[S3] raw ya existe, skip: {raw_path}")

    run([
        PYTHON, str(PROJECT_ROOT / "s3_model_code" / "parse_s3_outputs.py"),
        "--input-path", str(raw_path),
        "--output-path", str(parsed_path),
    ], "S3 parse")

    run([
        PYTHON, str(PROJECT_ROOT / "s3_model_code" / "evaluate_s3_answers.py"),
        "--input-path", str(parsed_path),
        "--output-path", str(results_path),
        "--summary-path", str(summary_path),
        "--group-summary-path", str(group_summary_path),
    ], "S3 evaluate")

    return results_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SYSTEM_RUNNERS = {
    "s1": run_s1,
    "s2": run_s2,
    "s3": run_s3,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Runner unificado: corre S1/S2/S3 sobre el set de evaluación."
    )
    parser.add_argument("--questions-path", type=Path, default=EVAL_QUESTIONS_PATH,
                        help="CSV de preguntas de evaluación.")
    parser.add_argument("--index-dir", type=Path, default=EVAL_INDEX_DIR,
                        help="Índice vectorial del corpus de evaluación.")
    parser.add_argument("--output-dir", type=Path, default=EVAL_OUTPUTS_DIR,
                        help="Directorio donde guardar todos los outputs.")
    parser.add_argument("--model", type=str, default=None,
                        help="Modelo LLM. Si se omite, usa el default de direct_llm.py.")
    parser.add_argument("--systems", type=str, default="s1,s2,s3",
                        help="Sistemas a correr, separados por coma. Default: s1,s2,s3.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Corre solo las primeras N preguntas. Para smoke tests.")
    parser.add_argument("--resume", action="store_true",
                        help="Retoma desde raw CSV existente sin repetir preguntas ya procesadas.")
    parser.add_argument("--force", action="store_true",
                        help="Sobreescribe outputs existentes, incluyendo el raw.")
    args = parser.parse_args()

    systems = [s.strip().lower() for s in args.systems.split(",")]
    invalid = [s for s in systems if s not in SYSTEM_RUNNERS]
    if invalid:
        print(f"Sistemas inválidos: {invalid}. Válidos: {list(SYSTEM_RUNNERS)}")
        sys.exit(1)

    if not args.questions_path.exists():
        print(f"No existe el archivo de preguntas: {args.questions_path}")
        print("Corré primero: python evaluation/build_eval_dataset.py")
        sys.exit(1)

    if not args.index_dir.exists():
        print(f"No existe el índice vectorial: {args.index_dir}")
        print("Corré primero: python evaluation/build_eval_dataset.py")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = model_tag(args.model)

    print(f"=== Pipeline de evaluación unificado ===")
    print(f"  Preguntas:  {args.questions_path}")
    print(f"  Índice:     {args.index_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Modelo:     {args.model or 'default'}")
    print(f"  Sistemas:   {systems}")
    print(f"  Limit:      {args.limit or 'todas'}")
    print(f"  Resume:     {args.resume}")
    print(f"  Force:      {args.force}")

    results: dict[str, Path] = {}
    for system in systems:
        print(f"\n{'='*50}")
        print(f"  Sistema: {system.upper()}")
        print(f"{'='*50}")
        runner = SYSTEM_RUNNERS[system]
        result_path = runner(
            questions_path=args.questions_path,
            index_dir=args.index_dir,
            output_dir=args.output_dir,
            model=args.model,
            limit=args.limit,
            resume=args.resume,
            force=args.force,
        )
        results[system] = result_path

    print("\n=== Pipeline completo ===")
    for system, path in results.items():
        print(f"  {system.upper()}: {path}")

    print(f"\nPara exportar XLSX corré:")
    print(f"  python evaluation/export_eval_results.py --input-dir {args.output_dir} --model {args.model or 'default'}")


if __name__ == "__main__":
    main()
