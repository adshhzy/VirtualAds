param(
    [string]$PythonPath = '',
    [string]$UvPath = '',
    [string]$EnvironmentPath = '',
    [string]$CachePath = ''
)

$ErrorActionPreference = 'Stop'
$projectPath = Split-Path -Parent $PSScriptRoot
$workspacePath = Split-Path -Parent $projectPath
$defaultEnvRoot = Join-Path $workspacePath 'env'
if (-not $PythonPath) { $PythonPath = Join-Path $defaultEnvRoot 'python312/python.exe' }
if (-not $UvPath) { $UvPath = Join-Path $defaultEnvRoot 'tools/bin/uv.exe' }
if (-not $EnvironmentPath) { $EnvironmentPath = Join-Path $defaultEnvRoot 'paste_gpu_performance' }
if (-not $CachePath) { $CachePath = Join-Path $defaultEnvRoot 'cache' }

# Supplied relative paths resolve against the caller's current directory.
$PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
$UvPath = (Resolve-Path -LiteralPath $UvPath).Path
$EnvironmentPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($EnvironmentPath)
$CachePath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($CachePath)
$lockPath = Join-Path $projectPath 'requirements-lock.txt'
$lockedRequirements = @(Get-Content -LiteralPath $lockPath | Where-Object { $_.Trim() -and -not $_.Trim().StartsWith('#') })
$torchRequirement = @($lockedRequirements | Where-Object { $_ -match '^torch==' })
if ($torchRequirement.Count -ne 1 -or $torchRequirement[0] -notmatch '\+cu128$') {
    throw 'requirements-lock.txt must contain exactly one torch==...+cu128 requirement.'
}

$previousUvCache = $env:UV_CACHE_DIR
try {
    $env:UV_CACHE_DIR = $CachePath
    $venvPython = Join-Path $EnvironmentPath 'Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        & $UvPath venv --python $PythonPath $EnvironmentPath
        if ($LASTEXITCODE -ne 0) { throw "uv venv failed ($LASTEXITCODE)." }
    }
    $venvPython = (Resolve-Path -LiteralPath $venvPython).Path
    # Install the CUDA wheel from PyTorch only; all other pinned packages use PyPI.
    & $UvPath pip install --python $venvPython --index-url 'https://download.pytorch.org/whl/cu128' --no-deps $torchRequirement[0]
    if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed ($LASTEXITCODE)." }
    $otherRequirements = @($lockedRequirements | Where-Object { $_ -notmatch '^torch==' })
    & $UvPath pip install --python $venvPython --index-url 'https://pypi.org/simple' --no-deps @otherRequirements
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed ($LASTEXITCODE)." }
    # Register the core package and legacy import aliases without changing locked dependencies.
    & $UvPath pip install --python $venvPython --no-deps --no-build-isolation --editable $projectPath
    if ($LASTEXITCODE -ne 0) { throw "Project installation failed ($LASTEXITCODE)." }
    & $UvPath pip check --python $venvPython
    if ($LASTEXITCODE -ne 0) { throw "Dependency validation failed ($LASTEXITCODE)." }
    & $venvPython -c 'import torch, triton, numpy, scipy; print(torch.__version__, triton.__version__, numpy.__version__, scipy.__version__); print(torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'
    if ($LASTEXITCODE -ne 0) { throw "Import validation failed ($LASTEXITCODE)." }
    Write-Host "Environment ready: $venvPython"
    Write-Host 'Run: .\scripts\run.ps1 shared --version'
} finally {
    $env:UV_CACHE_DIR = $previousUvCache
}
