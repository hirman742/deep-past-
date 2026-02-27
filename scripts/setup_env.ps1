param(
    [string]$CondaHome = "$env:USERPROFILE\miniforge3",
    [string]$EnvName = "deeppast-cleaning",
    [ValidateSet("gpu", "cpu")]
    [string]$ComputeTarget = "gpu"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$envFile = Join-Path $repoRoot "env.yml"
$condaBat = Join-Path $CondaHome "condabin\conda.bat"

if (-not (Test-Path $envFile)) {
    throw "Missing env file: $envFile"
}

if (-not (Test-Path $condaBat)) {
    $installer = Join-Path $env:TEMP "Miniforge3-Windows-x86_64.exe"
    $url = "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe"
    Write-Host "Downloading Miniforge installer..."
    Invoke-WebRequest -Uri $url -OutFile $installer

    Write-Host "Installing Miniforge to $CondaHome ..."
    $args = @(
        "/InstallationType=JustMe",
        "/RegisterPython=0",
        "/S",
        "/D=$CondaHome"
    )
    $p = Start-Process -FilePath $installer -ArgumentList $args -Wait -PassThru
    if ($p.ExitCode -ne 0) {
        throw "Miniforge installation failed with exit code $($p.ExitCode)."
    }
}

if (-not (Test-Path $condaBat)) {
    throw "conda not found after installation: $condaBat"
}

Write-Host "Creating/updating conda environment from $envFile ..."
& $condaBat env update -n $EnvName -f $envFile --prune
if ($LASTEXITCODE -ne 0) {
    throw "Conda environment update failed."
}

if ($ComputeTarget -eq "gpu") {
    Write-Host "Aligning PyTorch to conda GPU build (CUDA 12.8)..."
    $pkgList = & $condaBat list -n $EnvName
    if ($pkgList -match "^pytorch-cpu\s") {
        & $condaBat remove -n $EnvName -y pytorch-cpu
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to remove pytorch-cpu before GPU install."
        }
    }
    & $condaBat install -n $EnvName -c conda-forge -y "cuda-version=12.8" "pytorch=2.10.0=cuda128_mkl_py311_ha828eda_302"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install CUDA-enabled PyTorch."
    }
} else {
    Write-Host "Aligning PyTorch to conda CPU build..."
    & $condaBat install -n $EnvName -c conda-forge -y pytorch-cpu
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install pytorch-cpu."
    }

    & $condaBat run -n $EnvName python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('torch') else 1)"
    if ($LASTEXITCODE -ne 0) {
        & $condaBat install -n $EnvName -c conda-forge -y --force-reinstall pytorch pytorch-cpu libtorch
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to repair PyTorch installation."
        }
    }
}

Write-Host "Validating environment..."
& $condaBat run -n $EnvName python -V
if ($LASTEXITCODE -ne 0) {
    throw "Python validation failed."
}

& $condaBat run -n $EnvName python -c "import pandas, numpy, yaml, transformers, torch; print('OK: core packages import', torch.__version__, 'cuda=', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) {
    throw "Package import validation failed."
}

if ($ComputeTarget -eq "gpu") {
    & $condaBat run -n $EnvName python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
    if ($LASTEXITCODE -ne 0) {
        throw "GPU mode requested, but torch.cuda.is_available() is False."
    }
}

Write-Host ""
Write-Host "Done."
Write-Host "To activate in a new shell:"
Write-Host "  $CondaHome\Scripts\activate"
Write-Host "  conda activate $EnvName"
