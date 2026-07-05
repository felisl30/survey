param(
  [ValidateSet("dry-run", "real")]
  [string]$Mode = "real",

  [int]$Limit = 0,

  [string]$Model = "",

  [string]$Systems = "s1,s2,s3",

  [switch]$BuildDataset,

  [switch]$Resume,

  [switch]$Force
)

$ErrorActionPreference = "Stop"

# Moverse a la raiz del proyecto (un nivel arriba de evaluation/)
Set-Location (Split-Path $PSScriptRoot -Parent)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Invoke-Step {
  param(
    [string]$Name,
    [string[]]$CommandArgs
  )
  Write-Host ""
  Write-Host "== $Name =="
  & python @CommandArgs
  if ($LASTEXITCODE -ne 0) {
    throw "Falló el paso '$Name' con código $LASTEXITCODE."
  }
}

# ---------------------------------------------------------------------------
# Validaciones previas
# ---------------------------------------------------------------------------

if ($Mode -eq "real" -and -not $env:OPENAI_API_KEY) {
  throw "Mode 'real' requiere OPENAI_API_KEY seteado en el entorno o en .env."
}

$ModelTag = if ($Model) { $Model -replace "[/\-\.]", "_" } else { "default" }

Write-Host ""
Write-Host "========================================"
Write-Host "  Pipeline de evaluación unificado S1/S2/S3"
Write-Host "========================================"
Write-Host "  Mode:    $Mode"
Write-Host "  Limit:   $(if ($Limit -gt 0) { $Limit } else { 'todas' })"
Write-Host "  Model:   $(if ($Model) { $Model } else { 'default' })"
Write-Host "  Systems: $Systems"
Write-Host "  Resume:  $($Resume.IsPresent)"
Write-Host "  Force:   $($Force.IsPresent)"
Write-Host ""

# ---------------------------------------------------------------------------
# Paso 0: Construir dataset de evaluación (opcional)
# ---------------------------------------------------------------------------

if ($BuildDataset) {
  $buildArgs = @("evaluation/build_eval_dataset.py")
  if ($Force) { $buildArgs += "--force" }
  if ($Mode -eq "dry-run") { $buildArgs += "--skip-index" }
  Invoke-Step "Build eval dataset (80 retrieve + 20 direct)" $buildArgs
}
else {
  if (-not (Test-Path "data/eval/questions_eval.csv")) {
    throw "No existe data/eval/questions_eval.csv. Corré con -BuildDataset para generarlo."
  }
  if (-not (Test-Path "indexes/eval/embeddings.npy")) {
    throw "No existe el índice indexes/eval/. Corré con -BuildDataset para generarlo."
  }
  Write-Host "[skip] Dataset e índice ya existen."
}

# ---------------------------------------------------------------------------
# Rutas de salida
# ---------------------------------------------------------------------------

$outDir   = "outputs/eval"
$qPath    = "data/eval/questions_eval.csv"
$idxDir   = "indexes/eval"

# ---------------------------------------------------------------------------
# Paso 1: Correr los sistemas (run_evaluation_pipeline.py)
# ---------------------------------------------------------------------------

$pipelineArgs = @(
  "evaluation/run_evaluation_pipeline.py",
  "--questions-path", $qPath,
  "--index-dir",      $idxDir,
  "--output-dir",     $outDir,
  "--systems",        $Systems
)

if ($Model)            { $pipelineArgs += @("--model", $Model) }
if ($Limit -gt 0)      { $pipelineArgs += @("--limit", "$Limit") }
if ($Resume)           { $pipelineArgs += "--resume" }
if ($Force)            { $pipelineArgs += "--force" }

if ($Mode -eq "dry-run") {
  Write-Host ""
  Write-Host "[dry-run] Comando que se correría (sin API calls):"
  Write-Host ("  python " + ($pipelineArgs -join " "))
  Write-Host ""
  Write-Host "Pasá -Mode real para ejecutar con la API."
  exit 0
}

Invoke-Step "Correr S1/S2/S3 sobre eval set" $pipelineArgs

# ---------------------------------------------------------------------------
# Paso 2: Exportar XLSX
# ---------------------------------------------------------------------------

$xlsxPath = "$outDir/evaluation_${ModelTag}.xlsx"

$exportArgs = @(
  "evaluation/export_eval_results.py",
  "--input-dir",      $outDir,
  "--model",          $ModelTag,
  "--systems",        $Systems,
  "--questions-path", $qPath,
  "--output",         $xlsxPath
)

Invoke-Step "Exportar XLSX unificado" $exportArgs

# ---------------------------------------------------------------------------
# Resumen final
# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "========================================"
Write-Host "  Pipeline completado"
Write-Host "========================================"

$systemList = $Systems -split ","
foreach ($sys in $systemList) {
  $sys = $sys.Trim()
  $summary = "$outDir/${sys}_${ModelTag}_summary.json"
  if (Test-Path $summary) {
    Write-Host "  $($sys.ToUpper()) summary: $summary"
  }
}

Write-Host "  XLSX:    $xlsxPath"
Write-Host ""
