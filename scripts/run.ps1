# Deliberately not an advanced script: unknown CLI flags remain in $args.
param([ValidateSet('shared', 'batch', 'dynamic', 'compare', 'backends', 'render-img-modes', 'uploads', 'prepare', 'profile-uploads', 'check-uploads', 'check-prepare', 'tune', 'check', 'check-backends', 'check-policy', 'check-streaming', 'live-example', 'render-img-example', 'check-render-img')][string]$Benchmark = 'shared')

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
    'backends' { 'benchmark_backend_comparison.py' }
    'render-img-modes' { 'benchmark_render_img_modes.py' }
    'uploads' { 'benchmark_upload_comparison.py' }
    'prepare' { 'benchmark_prepare_comparison.py' }
    'profile-uploads' { 'profile_upload_stages.py' }
    'check-uploads' { 'check_upload_streaming.py' }
    'check-prepare' { 'check_prepare_streaming.py' }
    'tune' { 'benchmark_blur_tuning.py' }
    'check' { 'test_dynamic_batch.py' }
    'check-backends' { 'check_backend_contracts.py' }
    'check-policy' { 'test_backend_policy.py' }
    'check-streaming' { 'check_streaming_stability.py' }
    'live-example' { 'live_render_example.py' }
    'render-img-example' { 'render_img_example.py' }
    'check-render-img' { 'check_render_img.py' }
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
