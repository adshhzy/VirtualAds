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
$benchmarkModule = switch ($Benchmark) {
    'shared' { 'benchmarks.benchmark_shared_paste' }
    'batch' { 'benchmarks.benchmark_paste_gpu_batch' }
    'dynamic' { 'benchmarks.benchmark_dynamic_batch' }
    'compare' { 'benchmarks.benchmark_dynamic_comparison' }
    'backends' { 'benchmarks.benchmark_backend_comparison' }
    'render-img-modes' { 'benchmarks.benchmark_render_img_modes' }
    'uploads' { 'benchmarks.benchmark_upload_comparison' }
    'prepare' { 'benchmarks.benchmark_prepare_comparison' }
    'profile-uploads' { 'tools.profile_upload_stages' }
    'check-uploads' { 'tools.check_upload_streaming' }
    'check-prepare' { 'tools.check_prepare_streaming' }
    'tune' { 'benchmarks.benchmark_blur_tuning' }
    'check' { 'tests.test_dynamic_batch' }
    'check-backends' { 'tools.check_backend_contracts' }
    'check-policy' { 'tests.test_backend_policy' }
    'check-streaming' { 'tools.check_streaming_stability' }
    'live-example' { 'examples.live_render_example' }
    'render-img-example' { 'examples.render_img_example' }
    'check-render-img' { 'tools.check_render_img' }
}
$previousTritonCache = $env:TRITON_CACHE_DIR
$previousTorchCache = $env:TORCHINDUCTOR_CACHE_DIR
$exitCode = 1
Push-Location -LiteralPath $projectPath
try {
    if (-not $env:TRITON_CACHE_DIR) { $env:TRITON_CACHE_DIR = Join-Path $environmentRoot 'cache/triton' }
    if (-not $env:TORCHINDUCTOR_CACHE_DIR) { $env:TORCHINDUCTOR_CACHE_DIR = Join-Path $environmentRoot 'cache/torchinductor' }
    & $pythonPath -u -m $benchmarkModule @args
    $exitCode = $LASTEXITCODE
} finally {
    $env:TRITON_CACHE_DIR = $previousTritonCache
    $env:TORCHINDUCTOR_CACHE_DIR = $previousTorchCache
    Pop-Location
}
exit $exitCode
