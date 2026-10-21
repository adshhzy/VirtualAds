# Deliberately not an advanced script: unknown CLI flags remain in $args.
param([ValidateSet('shared', 'batch', 'dynamic', 'compare', 'tune', 'check')][string]$Benchmark = 'shared')

$ErrorActionPreference = 'Stop'
$projectPath = Split-Path -Parent $PSScriptRoot
$workspacePath = Split-Path -Parent $projectPath
$environmentRoot = Join-Path $workspacePath 'env'
$pythonCandidate = if ($env:PASTE_GPU_PYTHON) { $env:PASTE_GPU_PYTHON } else { Join-Path $environmentRoot 'paste_gpu_performance/Scripts/python.exe' }
if (-not (Test-Path -LiteralPath $pythonCandidate -PathType Leaf)) {
    throw "Python was not found at $pythonCandidate. Run scripts/setup.ps1 or set PASTE_GPU_PYTHON."
}
$pythonPath = (Resolve-Path -LiteralPath $pythonCandidate).Path
$benchmarkFile = switch ($Benchmark) {
    'shared' { 'benchmark_shared_paste.py' }
    'batch' { 'benchmark_paste_gpu_batch.py' }
    'dynamic' { 'benchmark_dynamic_batch.py' }
    'compare' { 'benchmark_dynamic_comparison.py' }
    'tune' { 'benchmark_blur_tuning.py' }
    'check' { 'test_dynamic_batch.py' }
}
$benchmarkPath = Join-Path $projectPath $benchmarkFile
$previousTritonCache = $env:TRITON_CACHE_DIR
$previousTorchCache = $env:TORCHINDUCTOR_CACHE_DIR
$exitCode = 1
Push-Location -LiteralPath $projectPath
try {
    if (-not $env:TRITON_CACHE_DIR) { $env:TRITON_CACHE_DIR = Join-Path $environmentRoot 'cache/triton' }
    if (-not $env:TORCHINDUCTOR_CACHE_DIR) { $env:TORCHINDUCTOR_CACHE_DIR = Join-Path $environmentRoot 'cache/torchinductor' }
    & $pythonPath -u $benchmarkPath @args
    $exitCode = $LASTEXITCODE
} finally {
    $env:TRITON_CACHE_DIR = $previousTritonCache
    $env:TORCHINDUCTOR_CACHE_DIR = $previousTorchCache
    Pop-Location
}
exit $exitCode
