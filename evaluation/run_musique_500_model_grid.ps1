param(
  [string]$Models = "gpt-5-nano,gpt-5-mini,gpt-4.1-mini",

  [string]$Systems = "s0,s1,s2,s3",

  [int]$Limit = 0,

  [switch]$BuildRag,

  [switch]$BuildIndex,

  [switch]$ForceIndex,

  [switch]$SkipSummary,

  [switch]$SkipPosthoc,

  [switch]$RunS4Focus,

  [int]$S4FocusLimit = 25
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

function Import-DotEnv {
  param([string]$Path)
  if (-not (Test-Path $Path)) { return }

  Get-Content $Path | ForEach-Object {
    $line = $_.Trim()
    if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) { return }
    $parts = $line.Split("=", 2)
    $name = $parts[0].Trim()
    $value = $parts[1].Trim().Trim('"').Trim("'")
    if ($name -and -not [Environment]::GetEnvironmentVariable($name, "Process")) {
      [Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
  }
}

function Invoke-Step {
  param(
    [string]$Name,
    [string[]]$CommandArgs
  )

  Write-Host ""
  Write-Host "================================================================================"
  Write-Host $Name
  Write-Host "================================================================================"
  Write-Host ("python " + ($CommandArgs -join " "))
  & python @CommandArgs
  if ($LASTEXITCODE -ne 0) {
    throw "Fallo el paso '$Name' con codigo $LASTEXITCODE."
  }
}

function Invoke-PowerShellStep {
  param(
    [string]$Name,
    [string[]]$CommandArgs
  )

  Write-Host ""
  Write-Host "================================================================================"
  Write-Host $Name
  Write-Host "================================================================================"
  Write-Host ("powershell " + ($CommandArgs -join " "))
  & powershell @CommandArgs
  if ($LASTEXITCODE -ne 0) {
    throw "Fallo el paso '$Name' con codigo $LASTEXITCODE."
  }
}

function Get-ModelTag {
  param([string]$Model)
  return ($Model -replace "[^A-Za-z0-9]+", "_")
}

Import-DotEnv ".env"
if (-not $env:OPENAI_API_KEY) {
  throw "Falta OPENAI_API_KEY en el entorno o en .env."
}

$modelList = $Models.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
$systemList = $Systems.Split(",") | ForEach-Object { $_.Trim().ToLower() } | Where-Object { $_ }

$questionsMc = "data/eval_mc/questions_musique_mc_500.csv"
$ragDir = "data/eval_mc/musique_mc_rag_500"
$ragQuestions = "$ragDir/questions.csv"
$ragCorpus = "$ragDir/corpus.csv"
$indexDir = "indexes/eval_mc/musique_mc_rag_500"
$baseOut = "outputs/eval_mc/musique_mc_rag_500"

if (-not (Test-Path $questionsMc)) {
  throw "No existe $questionsMc."
}

if ($BuildRag -or -not (Test-Path $ragQuestions) -or -not (Test-Path $ragCorpus)) {
  Invoke-Step "Build MuSiQue-500 RAG dataset" @(
    "evaluation/build_mc_rag_dataset.py",
    "--input-path", $questionsMc,
    "--output-dir", $ragDir,
    "--benchmark-name", "musique_mc_500",
    "--expected-n", "500"
  )
}
else {
  Write-Host "[skip] RAG dataset ya existe: $ragDir"
}

$indexFiles = @("$indexDir/chunks.csv", "$indexDir/embeddings.npy", "$indexDir/metadata.json")
$indexMissing = $false
foreach ($p in $indexFiles) {
  if (-not (Test-Path $p)) { $indexMissing = $true }
}

if ($BuildIndex -or $ForceIndex -or $indexMissing) {
  $indexArgs = @(
    "evaluation/build_mc_rag_index.py",
    "--corpus-path", $ragCorpus,
    "--output-dir", $indexDir
  )
  if ($ForceIndex) { $indexArgs += "--force" }
  Invoke-Step "Build MuSiQue-500 retrieval index" $indexArgs
}
else {
  Write-Host "[skip] Indice ya existe: $indexDir"
}

$limitArgs = @()
if ($Limit -gt 0) {
  $limitArgs = @("--limit", "$Limit")
}

foreach ($model in $modelList) {
  $tag = Get-ModelTag $model
  $outDir = "$baseOut/$tag"
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null

  Write-Host ""
  Write-Host "################################################################################"
  Write-Host "MODEL: $model"
  Write-Host "TAG:   $tag"
  Write-Host "OUT:   $outDir"
  Write-Host "################################################################################"

  if ($systemList -contains "s0") {
    $prefix = "s0_${tag}_musique_500"
    $s0Args = @(
      "-ExecutionPolicy", "Bypass",
      "-File", "./evaluation/run_s0_mc_pipeline.ps1",
      "-Mode", "real",
      "-Model", $model,
      "-QuestionPath", $questionsMc,
      "-OutputPrefix", $prefix,
      "-Resume"
    )
    if ($Limit -gt 0) { $s0Args += @("-Limit", "$Limit") }
    Invoke-PowerShellStep "Run S0 direct: $model" $s0Args
  }

  if ($systemList -contains "s1") {
    Invoke-Step "Run S1 RAG top-5: $model" (@(
      "modelos/s1/run_s1_mc_rag.py",
      "--questions-path", $ragQuestions,
      "--index-dir", $indexDir,
      "--output-path", "$outDir/s1_raw.csv",
      "--model", $model,
      "--top-k", "5",
      "--save-every", "1",
      "--resume"
    ) + $limitArgs)

    Invoke-Step "Parse S1: $model" @("modelos/s0/parse_s0_outputs.py", "--input-path", "$outDir/s1_raw.csv", "--output-path", "$outDir/s1_parsed.csv")
    Invoke-Step "Evaluate S1: $model" @("modelos/s0/evaluate_s0.py", "--input-path", "$outDir/s1_parsed.csv", "--output-path", "$outDir/s1_evaluated.csv", "--summary-path", "$outDir/s1_summary.json", "--group-summary-path", "$outDir/s1_group_summary.csv")
  }

  if ($systemList -contains "s2") {
    Invoke-Step "Run S2 adaptive RAG: $model" (@(
      "modelos/s2/run_s2_mc_real_adaptive.py",
      "--questions-path", $ragQuestions,
      "--index-dir", $indexDir,
      "--output-path", "$outDir/s2_raw.csv",
      "--model", $model,
      "--top-k", "5",
      "--threshold", "0.45",
      "--min-gap", "0.05",
      "--save-every", "1",
      "--resume"
    ) + $limitArgs)

    Invoke-Step "Parse S2: $model" @("modelos/s0/parse_s0_outputs.py", "--input-path", "$outDir/s2_raw.csv", "--output-path", "$outDir/s2_parsed.csv")
    Invoke-Step "Evaluate S2: $model" @("modelos/s0/evaluate_s0.py", "--input-path", "$outDir/s2_parsed.csv", "--output-path", "$outDir/s2_evaluated.csv", "--summary-path", "$outDir/s2_summary.json", "--group-summary-path", "$outDir/s2_group_summary.csv")
  }

  if ($systemList -contains "s3") {
    Invoke-Step "Run S3 MC FLARE-like: $model" (@(
      "modelos/s3/run_s3_mc_flare_like.py",
      "--questions-path", $ragQuestions,
      "--index-dir", $indexDir,
      "--output-path", "$outDir/s3_raw.csv",
      "--model", $model,
      "--top-k", "5",
      "--save-every", "1",
      "--resume"
    ) + $limitArgs)

    Invoke-Step "Parse S3: $model" @("modelos/s0/parse_s0_outputs.py", "--input-path", "$outDir/s3_raw.csv", "--output-path", "$outDir/s3_parsed.csv")
    Invoke-Step "Evaluate S3: $model" @("modelos/s0/evaluate_s0.py", "--input-path", "$outDir/s3_parsed.csv", "--output-path", "$outDir/s3_evaluated.csv", "--summary-path", "$outDir/s3_summary.json", "--group-summary-path", "$outDir/s3_group_summary.csv")
  }
}

if (-not $SkipSummary) {
  $summaryArgs = @(
    "evaluation/summarize_musique_500_model_grid.py",
    "--models", ($modelList -join ","),
    "--systems", ($systemList -join ","),
    "--expected-n", $(if ($Limit -gt 0) { "$Limit" } else { "500" })
  )
  Invoke-Step "Summarize MuSiQue-500 model grid" $summaryArgs
}

if (-not $SkipPosthoc) {
  Invoke-Step "Build S5 post-hoc router table" @(
    "evaluation/build_musique_500_s5_router.py",
    "--models", ($modelList -join ",")
  )

  Invoke-Step "Build S4 disagreement grid" @(
    "evaluation/build_musique_500_s4_disagreement_grid.py",
    "--models", ($modelList -join ","),
    "--s4-per-model", "$S4FocusLimit"
  )

  if ($RunS4Focus) {
    Invoke-Step "Run S4 FIRE-like on disagreement focus" @(
      "modelos/s4/run_s4_fire_like.py",
      "--input-path", "outputs/eval_mc/musique_mc_rag_500/posthoc/s4_focus_input.csv",
      "--index-dir", $indexDir,
      "--output-path", "outputs/eval_mc/musique_mc_rag_500/posthoc/s4_focus_raw.csv",
      "--source-system", "manual",
      "--limit", "$S4FocusLimit",
      "--use-index",
      "--initial-evidence-mode", "auto",
      "--claim-strategy", "rules",
      "--verification-strategy", "rules",
      "--query-strategy", "rules",
      "--repair-strategy", "rules",
      "--model", $modelList[0],
      "--resume"
    )
  }
}

Write-Host ""
Write-Host "DONE. Resumen:"
Write-Host "  outputs/eval_mc/musique_mc_rag_500/model_grid_summary/model_grid_metrics.csv"
Write-Host "  outputs/eval_mc/musique_mc_rag_500/model_grid_summary/model_grid_metrics.md"
Write-Host "  outputs/eval_mc/musique_mc_rag_500/posthoc/s5_policy_summary.csv"
Write-Host "  outputs/eval_mc/musique_mc_rag_500/posthoc/s4_disagreement_grid.csv"
Write-Host "  outputs/eval_mc/musique_mc_rag_500/posthoc/s4_focus_input.csv"
