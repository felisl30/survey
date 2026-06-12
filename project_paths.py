#!/usr/bin/env python3
"""
Canonical project paths for the survey experiments.

The repository is organized by experimental system:

- S0: direct LLM baseline.
- S1: basic RAG over HotpotQA-mini.
- S2: Adaptive-RAG.
- S3: FLARE-like active retrieval.
- S4: FIRE-like claim verification.

Inputs live under data/, vector indexes under indexes/, and generated artifacts
under outputs/. Scripts should import these constants instead of repeating path
strings so future folder changes stay easy to audit.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = Path("data")
INDEXES_DIR = Path("indexes")
OUTPUTS_DIR = Path("outputs")

S0_DATA_DIR = DATA_DIR
S0_OUTPUTS_DIR = OUTPUTS_DIR / "s0"

S1_DATA_DIR = DATA_DIR / "s1" / "hotpotqa_mini"
S1_INDEX_DIR = INDEXES_DIR / "s1" / "hotpotqa_mini"
S1_OUTPUTS_DIR = OUTPUTS_DIR / "s1"
S1_GENERATION_DIR = S1_OUTPUTS_DIR / "generation"
S1_RETRIEVAL_DIR = S1_OUTPUTS_DIR / "retrieval"
S1_EVALUATION_DIR = S1_OUTPUTS_DIR / "evaluation"

S2_DATA_DIR = DATA_DIR / "s2" / "adaptive_rag"
S2_INDEX_DIR = INDEXES_DIR / "s2" / "adaptive_rag"
S2_OUTPUTS_DIR = OUTPUTS_DIR / "s2"
S2_GENERATION_DIR = S2_OUTPUTS_DIR / "generation"
S2_ROUTING_DIR = S2_OUTPUTS_DIR / "routing"
S2_EVALUATION_DIR = S2_OUTPUTS_DIR / "evaluation"

HOTPOTQA_DISTRACTOR_DIR = DATA_DIR / "hotpotqa_distractor"

S0_MMLU_PATH = S0_DATA_DIR / "mmlu_50_questions.csv"
S0_TRUTHFULQA_PATH = S0_DATA_DIR / "questions_truthfulqa_open.csv"
S0_QUESTIONS_PATH = S0_DATA_DIR / "questions_s0.csv"
S0_RAW_OUTPUT_PATH = S0_OUTPUTS_DIR / "results_s0_raw.csv"
S0_PARSED_OUTPUT_PATH = S0_OUTPUTS_DIR / "results_s0_parsed.csv"
S0_EVALUATED_OUTPUT_PATH = S0_OUTPUTS_DIR / "results_s0_evaluated.csv"
S0_SUMMARY_PATH = S0_OUTPUTS_DIR / "evaluation_summary.json"
S0_GROUP_SUMMARY_PATH = S0_OUTPUTS_DIR / "evaluation_summary_by_group.csv"

S1_SOURCE_JSONL_PATH = HOTPOTQA_DISTRACTOR_DIR / "hotpotqa_distractor_validation.jsonl"
S1_QUESTIONS_PATH = S1_DATA_DIR / "questions_s1.csv"
S1_CORPUS_PATH = S1_DATA_DIR / "corpus_s1.csv"
S1_QRELS_PATH = S1_DATA_DIR / "qrels_s1.csv"
S1_RAW_OUTPUT_PATH = S1_GENERATION_DIR / "hotpotqa_mini_s1_raw.csv"
S1_PARSED_OUTPUT_PATH = S1_GENERATION_DIR / "hotpotqa_mini_s1_parsed.csv"
S1_ANSWER_RESULTS_PATH = S1_EVALUATION_DIR / "hotpotqa_mini_s1_answer_results.csv"
S1_ANSWER_SUMMARY_PATH = S1_EVALUATION_DIR / "hotpotqa_mini_s1_answer_summary.json"
S1_ANSWER_GROUP_SUMMARY_PATH = S1_EVALUATION_DIR / "hotpotqa_mini_s1_answer_summary_by_group.csv"
S1_SELECTED_TOP_K_PATH = S1_OUTPUTS_DIR / "selected_top_k.txt"

S2_QUESTIONS_PATH = S2_DATA_DIR / "questions_s2.csv"
S2_CORPUS_PATH = S2_DATA_DIR / "corpus_s2.csv"
S2_QRELS_PATH = S2_DATA_DIR / "qrels_s2.csv"
S2_SUMMARY_PATH = S2_DATA_DIR / "summary_s2.json"
S2_RAW_OUTPUT_PATH = S2_GENERATION_DIR / "adaptive_rag_s2_raw.csv"
S2_PARSED_OUTPUT_PATH = S2_GENERATION_DIR / "adaptive_rag_s2_parsed.csv"
S2_ROUTER_RESULTS_PATH = S2_ROUTING_DIR / "router_s2_results.csv"
S2_ROUTER_SUMMARY_PATH = S2_ROUTING_DIR / "router_s2_summary.json"
S2_ROUTING_RESULTS_PATH = S2_EVALUATION_DIR / "adaptive_rag_s2_routing_results.csv"
S2_ROUTING_SUMMARY_PATH = S2_EVALUATION_DIR / "adaptive_rag_s2_routing_summary.json"
S2_ROUTING_GROUP_SUMMARY_PATH = S2_EVALUATION_DIR / "adaptive_rag_s2_routing_summary_by_group.csv"
S2_ANSWER_RESULTS_PATH = S2_EVALUATION_DIR / "adaptive_rag_s2_answer_results.csv"
S2_ANSWER_SUMMARY_PATH = S2_EVALUATION_DIR / "adaptive_rag_s2_answer_summary.json"
S2_ANSWER_GROUP_SUMMARY_PATH = S2_EVALUATION_DIR / "adaptive_rag_s2_answer_summary_by_group.csv"

S3_OUTPUTS_DIR = OUTPUTS_DIR / "s3"
S3_GENERATION_DIR = S3_OUTPUTS_DIR / "generation"
S3_EVALUATION_DIR = S3_OUTPUTS_DIR / "evaluation"
S3_RAW_OUTPUT_PATH = S3_GENERATION_DIR / "flare_like_s3_raw.csv"
S3_PARSED_OUTPUT_PATH = S3_GENERATION_DIR / "flare_like_s3_parsed.csv"
S3_ANSWER_RESULTS_PATH = S3_EVALUATION_DIR / "flare_like_s3_answer_results.csv"
S3_ANSWER_SUMMARY_PATH = S3_EVALUATION_DIR / "flare_like_s3_answer_summary.json"
S3_ANSWER_GROUP_SUMMARY_PATH = S3_EVALUATION_DIR / "flare_like_s3_answer_summary_by_group.csv"

S4_OUTPUTS_DIR = OUTPUTS_DIR / "s4"
S4_GENERATION_DIR = S4_OUTPUTS_DIR / "generation"
S4_EVALUATION_DIR = S4_OUTPUTS_DIR / "evaluation"
S4_VERIFICATION_DIR = S4_OUTPUTS_DIR / "verification"
S4_COMPARISON_DIR = S4_OUTPUTS_DIR / "comparison"
S4_RAW_OUTPUT_PATH = S4_GENERATION_DIR / "fire_like_s4_raw.csv"
S4_PARSED_OUTPUT_PATH = S4_GENERATION_DIR / "fire_like_s4_parsed.csv"
S4_CLAIM_RESULTS_PATH = S4_EVALUATION_DIR / "fire_like_s4_claim_results.csv"
S4_CLAIM_SUMMARY_PATH = S4_EVALUATION_DIR / "fire_like_s4_claim_summary.json"
S4_CLAIM_GROUP_SUMMARY_PATH = S4_EVALUATION_DIR / "fire_like_s4_claim_summary_by_group.csv"
S4_ANSWER_RESULTS_PATH = S4_EVALUATION_DIR / "fire_like_s4_answer_results.csv"
S4_ANSWER_SUMMARY_PATH = S4_EVALUATION_DIR / "fire_like_s4_answer_summary.json"
S4_ANSWER_GROUP_SUMMARY_PATH = S4_EVALUATION_DIR / "fire_like_s4_answer_summary_by_group.csv"
S4_VERIFY_PREFIX = "verify_s4"
