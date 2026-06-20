param(
  [ValidateSet("dry-run", "real")]
  [string]$Mode = "dry-run",

  [int]$Limit = 0,

  [string]$Model = "",

  [switch]$BuildDataset,

  [ValidateSet("dry-run", "real")]
  [string]$BuildMode = "dry-run",

  [string]$Datasets = "hotpotqa,musique,2wiki",

  [int]$PerDataset = 30,

  [string]$GeneratorModel = "gpt-5-mini",

  [switch]$BuildOnly,

  [string]$QuestionPath = "",

  [string]$OutputPrefix = "",

  [switch]$Resume,

  [switch]$Force
)

$ErrorActionPreference = "Stop"

Set-Location (Split-Path $PSScriptRoot -Parent)

function Invoke-Step {
  param(
    [string]$Name,
    [string[]]$CommandArgs
  )
  Write-Host ""
  Write-Host "== $Name =="
  & python @CommandArgs
  if ($LASTEXITCODE -ne 0) {
    throw "Fallo el paso '$Name' con codigo $LASTEXITCODE."
  }
}

function Get-ModelTag {
  param([string]$Name)
  if ($Name) {
    return ($Name -replace "[/\-\.]", "_")
  }
  return "default"
}

function Import-DotEnv {
  param([string]$Path)
  if (-not (Test-Path $Path)) {
    return
  }

  Get-Content $Path | ForEach-Object {
    $line = $_.Trim()
    if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) {
      return
    }

    $parts = $line.Split("=", 2)
    $name = $parts[0].Trim()
    $value = $parts[1].Trim().Trim('"').Trim("'")
    if ($name -and -not [Environment]::GetEnvironmentVariable($name, "Process")) {
      [Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
  }
}

Import-DotEnv ".env"

$needsS0Api = ($Mode -eq "real" -and -not $BuildOnly)
$needsBuilderApi = ($Mode -eq "real" -and $BuildDataset -and $BuildMode -eq "real")
if (($needsS0Api -or $needsBuilderApi) -and -not $env:OPENAI_API_KEY) {
  throw "La corrida real requiere OPENAI_API_KEY seteado en el entorno o en .env."
}

$modelTag = Get-ModelTag $Model
$outDir = "outputs/eval_mc"
$qPath = if ($QuestionPath) { $QuestionPath } else { "data/eval_mc/questions_mc_eval.csv" }
$prefix = if ($OutputPrefix) { $OutputPrefix } else { "s0_${modelTag}" }
$rawPath = "$outDir/${prefix}_raw.csv"
$parsedPath = "$outDir/${prefix}_parsed.csv"
$resultsPath = "$outDir/${prefix}_mc_results.csv"
$summaryPath = "$outDir/${prefix}_mc_summary.csv"

Write-Host ""
Write-Host "========================================"
Write-Host "  S0 direct baseline sobre benchmark MC"
Write-Host "========================================"
Write-Host "  Mode:         $Mode"
Write-Host "  Model:        $(if ($Model) { $Model } else { 'default' })"
Write-Host "  Limit:        $(if ($Limit -gt 0) { $Limit } else { 'todas' })"
Write-Host "  BuildDataset: $($BuildDataset.IsPresent)"
Write-Host "  BuildMode:    $BuildMode"
Write-Host "  Datasets:     $Datasets"
Write-Host "  PerDataset:   $PerDataset"
Write-Host "  BuildOnly:    $($BuildOnly.IsPresent)"
Write-Host "  QuestionPath: $qPath"
Write-Host "  OutputPrefix: $prefix"
Write-Host "  Resume:       $($Resume.IsPresent)"
Write-Host "  Force:        $($Force.IsPresent)"
Write-Host ""

if ($BuildDataset) {
  $buildArgs = @(
    "evaluation/build_mc_eval_dataset.py",
    "--mode", $BuildMode,
    "--datasets", $Datasets,
    "--per-dataset", "$PerDataset"
  )
  if ($GeneratorModel) { $buildArgs += @("--generator-model", $GeneratorModel) }
  if ($Force) { $buildArgs += "--force" }

  if ($Mode -eq "dry-run") {
    Write-Host "[dry-run] Comando para construir dataset:"
    Write-Host ("  python " + ($buildArgs -join " "))
  }
  else {
    Invoke-Step "Build MC eval dataset" $buildArgs
  }
}
elseif (-not (Test-Path $qPath)) {
  if ($Mode -eq "dry-run") {
    Write-Host "[dry-run] Aviso: todavia no existe $qPath. Agrega -BuildDataset para generarlo."
  }
  else {
    throw "No existe $qPath. Corre con -BuildDataset para generarlo."
  }
}

if ($BuildOnly) {
  Write-Host ""
  Write-Host "BuildOnly activo: no se corre S0."
  exit 0
}

$runArgs = @(
  "run_s0_direct.py",
  "--input-path", $qPath,
  "--output-path", $rawPath
)
if ($Model) { $runArgs += @("--model", $Model) }
if ($Limit -gt 0) { $runArgs += @("--limit", "$Limit") }
if ($Resume) { $runArgs += "--resume" }

$parseArgs = @(
  "parse_s0_outputs.py",
  "--input-path", $rawPath,
  "--output-path", $parsedPath
)

$evalArgs = @(
  "evaluation/evaluate_mc_accuracy.py",
  "--input-path", $parsedPath,
  "--output-path", $resultsPath,
  "--summary-path", $summaryPath
)

if ($Mode -eq "dry-run") {
  Write-Host ""
  Write-Host "[dry-run] Comandos que se correrian sin gastar API:"
  Write-Host ("  python " + ($runArgs -join " "))
  Write-Host ("  python " + ($parseArgs -join " "))
  Write-Host ("  python " + ($evalArgs -join " "))
  Write-Host ""
  Write-Host "Para ejecutar de verdad: usa -Mode real."
  exit 0
}

Invoke-Step "Run S0 direct" $runArgs
Invoke-Step "Parse S0 outputs" $parseArgs
Invoke-Step "Evaluate S0 accuracy" $evalArgs

Write-Host ""
Write-Host "========================================"
Write-Host "  Pipeline S0 MC completado"
Write-Host "========================================"
Write-Host "  Raw:           $rawPath"
Write-Host "  Parsed:        $parsedPath"
Write-Host "  Results:       $resultsPath"
Write-Host "  Summary:       $summaryPath"
Write-Host ""
