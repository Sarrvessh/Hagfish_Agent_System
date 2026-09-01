$ErrorActionPreference = "Stop"

# Overnight benchmark runner for simple-hpo-bench on Windows PowerShell.
#
# Original requested dataset names:
# australian car phoneme vehicle blood-transfusion breast-cancer jasmine sylvine
#
# This script normalizes common variants and skips datasets unavailable in the
# currently installed simple-hpo-bench package.

$Datasets = @(
    "australian",
    "car",
    "phoneme",
    "vehicle",
    "blood-transfusion",
    "breast-cancer",
    "jasmine",
    "sylvine"
)

$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$OutputCsv = Join-Path $RepositoryRoot "results/hpobench/results_merged.csv"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "[FATAL] python is not available in PATH" -ForegroundColor Red
    exit 1
}

# Query supported datasets once from simple-hpo-bench.
$availableRaw = python -c "from hpo_benchmarks import HPOBench; print('\n'.join(HPOBench.available_dataset_names))"
if (-not $availableRaw) {
    Write-Host "[FATAL] Could not determine available datasets from simple-hpo-bench" -ForegroundColor Red
    exit 1
}

$Available = @{}
$availableRaw -split "`r?`n" | ForEach-Object {
    if ($_ -and $_.Trim().Length -gt 0) {
        $Available[$_.Trim()] = $true
    }
}

function Normalize-DatasetName {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Requested
    )

    switch ($Requested) {
        "blood-transfusion" { return "blood_transfusion" }
        "breast-cancer" { return "breast_cancer" }
        default { return $Requested.Replace("-", "_") }
    }
}

$SuccessCount = 0
$SkippedCount = 0
$FailedCount = 0

foreach ($datasetRequested in $Datasets) {
    $dataset = Normalize-DatasetName -Requested $datasetRequested
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

    Write-Host "[$timestamp] Starting dataset: $datasetRequested (normalized: $dataset)"

    if (-not $Available.ContainsKey($dataset)) {
        Write-Host "[WARN] Dataset '$dataset' is not available in this simple-hpo-bench install. Skipping." -ForegroundColor Yellow
        $SkippedCount++
        Start-Sleep -Seconds 5
        continue
    }

    python (Join-Path $PSScriptRoot "run_all_hpobench.py") `
        --dataset-name $dataset `
        --stages optuna hagfish `
        --seeds 0 1 2 3 4 5 6 7 8 9 `
        --optuna-n-trials 150 `
        --hagfish-n-trials 150 `
        --num-fidelity-steps 50 `
        --append-existing `
        --output-csv $OutputCsv

    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Dataset '$dataset' completed." -ForegroundColor Green
        $SuccessCount++
    }
    else {
        Write-Host "[ERROR] Dataset '$dataset' failed with exit code $LASTEXITCODE. Continuing to next dataset." -ForegroundColor Red
        $FailedCount++
    }

    Start-Sleep -Seconds 5
}

Write-Host ""
Write-Host "========== OVERNIGHT RUN SUMMARY =========="
Write-Host "Successful datasets: $SuccessCount"
Write-Host "Skipped datasets:    $SkippedCount"
Write-Host "Failed datasets:     $FailedCount"
Write-Host "Merged CSV:          $OutputCsv"
