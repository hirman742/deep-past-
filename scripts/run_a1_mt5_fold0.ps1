param(
    [string]$EnvName = "deeppast-cleaning",
    [string]$Config = "configs/mt5_small_lora_8gb.yaml",
    [int]$Fold = 0
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$condaBat = "$env:USERPROFILE\miniforge3\condabin\conda.bat"

if (-not (Test-Path $condaBat)) {
    throw "conda not found at $condaBat"
}

Set-Location $repoRoot

& $condaBat run -n $EnvName python scripts/preprocess.py --config $Config
& $condaBat run -n $EnvName python scripts/train_mt5_lora.py --config $Config --fold $Fold
& $condaBat run -n $EnvName python scripts/infer_mt5_lora.py --config $Config --fold $Fold

Write-Host "Done A1 fold $Fold pipeline."
