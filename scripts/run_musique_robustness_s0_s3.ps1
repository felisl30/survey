param(
  [string]$Model = "gpt-5.4-mini",
  [string]$BaseOut = "outputs/eval_mc/robustness_musique",
  [Nullable[int]]$Limit = $null
)

$ErrorActionPreference = "Stop"

function Get-ModelTag([string]$Name) {
  return $Name.Replace(".", "_").Replace("-", "_")
}

function Invoke-Step([string]$Name, [string[]]$ArgsList) {
  Write-Host ""
  Write-Host "---- $Name ----"
  & python @ArgsList
  if ($LASTEXITCODE -ne 0) {
    throw "Falló: $Name"
  }
}

function Invoke-EvalMc([string]$InputPath, [string]$Prefix) {
  Invoke-Step "Parse $Prefix" @(
    "modelos/s0/parse_s0_outputs.py",
    "--input-path", $InputPath,
    "--output-path", "${Prefix}_parsed.csv"
  )

  Invoke-Step "Evaluate $Prefix" @(
    "modelos/s0/evaluate_s0.py",
    "--input-path", "${Prefix}_parsed.csv",
    "--output-path", "${Prefix}_evaluated.csv",
    "--summary-path", "${Prefix}_summary.json",
    "--group-summary-path", "${Prefix}_group_summary.csv"
  )
}

$questions = "data/eval_mc/robustness_musique/questions.csv"
$conditions = @("clean", "noisy", "adversarial")
$tag = Get-ModelTag $Model
$out = Join-Path $BaseOut $tag
New-Item -ItemType Directory -Force -Path $out | Out-Null

$limitArgs = @()
if ($null -ne $Limit) {
  $limitArgs = @("--limit", [string]$Limit)
}

Write-Host ""
Write-Host "================================================================================"
Write-Host "ROBUSTNESS RUN"
Write-Host "MODEL: $Model"
Write-Host "TAG:   $tag"
Write-Host "OUT:   $out"
if ($null -ne $Limit) {
  Write-Host "LIMIT: $Limit"
} else {
  Write-Host "LIMIT: full"
}
Write-Host "================================================================================"

Invoke-Step "S0 direct baseline" (@(
  "modelos/s0/run_s0_direct.py",
  "--input-path", $questions,
  "--output-path", "$out/s0_raw.csv",
  "--model", $Model,
  "--save-every", "1",
  "--resume"
) + $limitArgs)

Invoke-EvalMc "$out/s0_raw.csv" "$out/s0"

foreach ($condition in $conditions) {
  $indexDir = "indexes/eval_mc/robustness_musique_$condition"
  $condOut = Join-Path $out $condition
  New-Item -ItemType Directory -Force -Path $condOut | Out-Null

  Write-Host ""
  Write-Host "================================================================================"
  Write-Host "CONDITION: $condition"
  Write-Host "INDEX:     $indexDir"
  Write-Host "OUT:       $condOut"
  Write-Host "================================================================================"

  Invoke-Step "S1 classic RAG top-5 $condition" (@(
    "modelos/s1/run_s1_mc_rag.py",
    "--questions-path", $questions,
    "--index-dir", $indexDir,
    "--output-path", "$condOut/s1_raw.csv",
    "--model", $Model,
    "--top-k", "5",
    "--save-every", "1",
    "--resume"
  ) + $limitArgs)

  Invoke-EvalMc "$condOut/s1_raw.csv" "$condOut/s1"

  Invoke-Step "S2 real adaptive $condition" (@(
    "modelos/s2/run_s2_mc_real_adaptive.py",
    "--questions-path", $questions,
    "--index-dir", $indexDir,
    "--output-path", "$condOut/s2_raw.csv",
    "--model", $Model,
    "--top-k", "5",
    "--threshold", "0.45",
    "--min-gap", "0.05",
    "--save-every", "1",
    "--resume"
  ) + $limitArgs)

  Invoke-EvalMc "$condOut/s2_raw.csv" "$condOut/s2"

  Invoke-Step "S3-MC FLARE-like $condition" (@(
    "modelos/s3/run_s3_mc_flare_like.py",
    "--questions-path", $questions,
    "--index-dir", $indexDir,
    "--output-path", "$condOut/s3_mc_raw.csv",
    "--model", $Model,
    "--top-k", "5",
    "--confidence-threshold", "0.78",
    "--score-threshold", "0.45",
    "--min-gap", "0.05",
    "--save-every", "1",
    "--resume"
  ) + $limitArgs)

  Invoke-EvalMc "$condOut/s3_mc_raw.csv" "$condOut/s3_mc"
}

Invoke-Step "Summarize robustness" @(
  "evaluation/summarize_musique_robustness_s0_s3.py",
  "--base-dir", $out
)

Invoke-Step "Deep robustness analysis" @(
  "evaluation/analyze_musique_robustness_deep.py",
  "--base-dir", $out
)

Invoke-Step "Export qualitative examples" @(
  "evaluation/export_musique_robustness_qualitative_examples.py",
  "--base-dir", $out
)

Write-Host ""
Write-Host "Robustness S0-S3 run finished."
Write-Host "Outputs: $out"
