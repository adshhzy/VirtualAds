# Run CPU contracts by default; CUDA tests require explicit PASTE_TEST_DEVICE=cuda:0.
# Optional arguments replace discovery, e.g. tests.test_render_api -v.
$ErrorActionPreference = 'Stop'
$projectPath = Split-Path -Parent $PSScriptRoot
$workspacePath = Split-Path -Parent $projectPath
$pythonCandidate = if ($env:PASTE_GPU_PYTHON) { $env:PASTE_GPU_PYTHON } else {
    Join-Path $workspacePath 'env/paste_gpu_performance/Scripts/python.exe'
}
if (-not (Test-Path -LiteralPath $pythonCandidate -PathType Leaf)) {
    throw 'Project Python was not found. Run scripts/setup.ps1 or set PASTE_GPU_PYTHON.'
}
$pythonPath = (Resolve-Path -LiteralPath $pythonCandidate).Path
$testArguments = if ($args.Count) { $args } else { @('discover', '-s', 'tests', '-t', '.') }
$exitCode = 1
Push-Location -LiteralPath $projectPath
try {
    & $pythonPath -B -m unittest @testArguments
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $exitCode
