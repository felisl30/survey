param(
  [ValidateSet("demo", "dry-run", "real")]
  [string]$Mode = "demo",

  [int]$Limit = 5,

  [string]$Tag = "",

  [switch]$PrepareData,

  [switch]$BuildIndexes
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
  param(
    [string]$Name,
    [string[]]$CommandArgs
  )

  Write-Host ""
  Write-Host "== $Name =="
  & python @CommandArgs
}

function Add-DryRunFlag {
  param([string[]]$Args)
  if ($Mode -eq "dry-run") {
    return $Args + "--dry-run"
  }
  return $Args
}

if (-not $Tag) {
  $Tag = if ($Mode -eq "demo") { "demo" } else { "${Mode}_${Limit}" }
}

if ($Mode -eq "real" -and -not $env:OPENAI_API_KEY) {
  throw "Mode real requiere OPENAI_API_KEY en el entorno o en .env."
}

Write-Host "Pipeline frame S1-S4"
Write-Host "Mode: $Mode"
Write-Host "Limit: $Limit"
Write-Host "Tag: $Tag"

if ($Mode -ne "demo") {
  if ($PrepareData) {
    Invoke-Step "Prepare S0 dataset" @("prepare_s0_dataset.py")
    Invoke-Step "Prepare S1 dataset" @("s1_model_code/prepare_hotpotqa_mini.py", "--per-topic", "1")
    Invoke-Step "Prepare S2 dataset" @(
      "s2_model_code/prepare_s2_dataset.py",
      "--n-direct-mmlu", "1",
      "--n-direct-truthfulqa", "1",
      "--n-retrieve", "1",
      "--n-abstain", "1",
      "--n-clarify", "1"
    )
  }

  if ($BuildIndexes) {
    Invoke-Step "Build S1 index" @("s1_model_code/build_s1_index.py")
    Invoke-Step "Build S2 index" @("s2_model_code/build_s2_index.py", "--overwrite")
  }
}

# ---------------------------------------------------------------------------
# S1
# ---------------------------------------------------------------------------

$s1Raw = "outputs/s1/generation/hotpotqa_mini_s1_raw_$Tag.csv"
$s1Parsed = "outputs/s1/generation/hotpotqa_mini_s1_parsed_$Tag.csv"
$s1Eval = "outputs/s1/evaluation/hotpotqa_mini_s1_answer_results_$Tag.csv"
$s1Summary = "outputs/s1/evaluation/hotpotqa_mini_s1_answer_summary_$Tag.json"
$s1Group = "outputs/s1/evaluation/hotpotqa_mini_s1_answer_summary_by_group_$Tag.csv"

if ($Mode -eq "demo") {
  $s1Raw = "outputs/s1/generation/hotpotqa_mini_s1_raw.csv"
}
else {
  $args = @(
    "s1_model_code/run_s1_rag.py",
    "--limit", "$Limit",
    "--output-path", $s1Raw
  )
  Invoke-Step "S1 run" (Add-DryRunFlag $args)
}

Invoke-Step "S1 parse" @(
  "s1_model_code/parse_s1_outputs.py",
  "--input-path", $s1Raw,
  "--output-path", $s1Parsed
)
Invoke-Step "S1 evaluate answers" @(
  "s1_model_code/evaluate_s1_answers.py",
  "--input-path", $s1Parsed,
  "--output-path", $s1Eval,
  "--summary-path", $s1Summary,
  "--group-summary-path", $s1Group
)

# ---------------------------------------------------------------------------
# S2
# ---------------------------------------------------------------------------

$s2Raw = "outputs/s2/generation/adaptive_rag_s2_raw_$Tag.csv"
$s2Parsed = "outputs/s2/generation/adaptive_rag_s2_parsed_$Tag.csv"
$s2RoutingEval = "outputs/s2/evaluation/adaptive_rag_s2_routing_results_$Tag.csv"
$s2RoutingSummary = "outputs/s2/evaluation/adaptive_rag_s2_routing_summary_$Tag.json"
$s2RoutingGroup = "outputs/s2/evaluation/adaptive_rag_s2_routing_summary_by_group_$Tag.csv"
$s2AnswerEval = "outputs/s2/evaluation/adaptive_rag_s2_answer_results_$Tag.csv"
$s2AnswerSummary = "outputs/s2/evaluation/adaptive_rag_s2_answer_summary_$Tag.json"
$s2AnswerGroup = "outputs/s2/evaluation/adaptive_rag_s2_answer_summary_by_group_$Tag.csv"

if ($Mode -eq "demo") {
  $s2Raw = "outputs/s2/generation/adaptive_rag_s2_raw_test_5.csv"
}
else {
  $args = @(
    "s2_model_code/run_s2_adaptive_rag.py",
    "--router-strategy", "rules",
    "--limit", "$Limit",
    "--output-path", $s2Raw
  )
  Invoke-Step "S2 run" (Add-DryRunFlag $args)
}

Invoke-Step "S2 parse" @(
  "s2_model_code/parse_s2_outputs.py",
  "--input-path", $s2Raw,
  "--output-path", $s2Parsed
)
Invoke-Step "S2 evaluate routing" @(
  "s2_model_code/evaluate_s2_routing.py",
  "--input-path", $s2Parsed,
  "--output-path", $s2RoutingEval,
  "--summary-path", $s2RoutingSummary,
  "--group-summary-path", $s2RoutingGroup
)
Invoke-Step "S2 evaluate answers" @(
  "s2_model_code/evaluate_s2_answers.py",
  "--input-path", $s2Parsed,
  "--output-path", $s2AnswerEval,
  "--summary-path", $s2AnswerSummary,
  "--group-summary-path", $s2AnswerGroup
)

# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------

$s3Raw = "outputs/s3/generation/flare_like_s3_raw_$Tag.csv"
$s3Parsed = "outputs/s3/generation/flare_like_s3_parsed_$Tag.csv"
$s3Eval = "outputs/s3/evaluation/flare_like_s3_answer_results_$Tag.csv"
$s3Summary = "outputs/s3/evaluation/flare_like_s3_answer_summary_$Tag.json"
$s3Group = "outputs/s3/evaluation/flare_like_s3_answer_summary_by_group_$Tag.csv"

if ($Mode -eq "demo") {
  $s3Raw = "outputs/s3/generation/flare_like_s3_raw_test_5_v4.csv"
}
else {
  $args = @(
    "s3_model_code/run_s3_flare_like.py",
    "--limit", "$Limit",
    "--max-steps", "3",
    "--top-k-per-step", "2",
    "--max-retrieval-steps", "2",
    "--max-total-chunks", "4",
    "--retrieval-strategy", "rules",
    "--output-path", $s3Raw
  )
  Invoke-Step "S3 run" (Add-DryRunFlag $args)
}

Invoke-Step "S3 parse" @(
  "s3_model_code/parse_s3_outputs.py",
  "--input-path", $s3Raw,
  "--output-path", $s3Parsed
)
Invoke-Step "S3 evaluate answers" @(
  "s3_model_code/evaluate_s3_answers.py",
  "--input-path", $s3Parsed,
  "--output-path", $s3Eval,
  "--summary-path", $s3Summary,
  "--group-summary-path", $s3Group
)

# ---------------------------------------------------------------------------
# S4
# ---------------------------------------------------------------------------

$s4Raw = "outputs/s4/generation/fire_like_s4_raw_$Tag.csv"
$s4Parsed = "outputs/s4/generation/fire_like_s4_parsed_$Tag.csv"
$s4ClaimEval = "outputs/s4/evaluation/fire_like_s4_claim_results_$Tag.csv"
$s4ClaimSummary = "outputs/s4/evaluation/fire_like_s4_claim_summary_$Tag.json"
$s4ClaimGroup = "outputs/s4/evaluation/fire_like_s4_claim_summary_by_group_$Tag.csv"
$s4AnswerEval = "outputs/s4/evaluation/fire_like_s4_answer_results_$Tag.csv"
$s4AnswerSummary = "outputs/s4/evaluation/fire_like_s4_answer_summary_$Tag.json"
$s4AnswerGroup = "outputs/s4/evaluation/fire_like_s4_answer_summary_by_group_$Tag.csv"

if ($Mode -eq "demo") {
  $s4Raw = "outputs/s4/generation/fire_like_s4_raw_retrieve_test_5_hybrid_claims_llm_verify.csv"
}
else {
  $args = @(
    "s4_model_code/run_s4_fire_like.py",
    "--input-path", $s2Parsed,
    "--limit", "$Limit",
    "--output-path", $s4Raw,
    "--claim-strategy", "rules",
    "--verification-strategy", "rules",
    "--query-strategy", "rules",
    "--repair-strategy", "rules",
    "--initial-evidence-mode", "auto"
  )
  if ($Mode -eq "dry-run") {
    $args = $args + "--dry-run" + "--no-index"
  }
  else {
    $args = $args + "--use-index"
  }
  Invoke-Step "S4 run" $args
}

Invoke-Step "S4 parse" @(
  "s4_model_code/parse_s4_outputs.py",
  "--input-path", $s4Raw,
  "--output-path", $s4Parsed
)
Invoke-Step "S4 evaluate claims" @(
  "s4_model_code/evaluate_s4_claims.py",
  "--input-path", $s4Parsed,
  "--claim-output-path", $s4ClaimEval,
  "--summary-path", $s4ClaimSummary,
  "--group-summary-path", $s4ClaimGroup
)
Invoke-Step "S4 evaluate answers" @(
  "s4_model_code/evaluate_s4_answers.py",
  "--input-path", $s4Parsed,
  "--output-path", $s4AnswerEval,
  "--summary-path", $s4AnswerSummary,
  "--group-summary-path", $s4AnswerGroup
)
Invoke-Step "S4 verify" @(
  "s4_model_code/verify_s4_results.py",
  "--s4-parsed-path", $s4Parsed,
  "--claim-results-path", $s4ClaimEval,
  "--answer-results-path", $s4AnswerEval,
  "--output-dir", "outputs/s4/verification",
  "--prefix", $Tag
)

Write-Host ""
Write-Host "== Pipeline finished =="
Write-Host "Metric summaries:"
Write-Host "- $s1Summary"
Write-Host "- $s2RoutingSummary"
Write-Host "- $s2AnswerSummary"
Write-Host "- $s3Summary"
Write-Host "- $s4ClaimSummary"
Write-Host "- $s4AnswerSummary"
Write-Host "- outputs/s4/verification/${Tag}_summary.txt"
