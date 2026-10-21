# Developer-only formatting and lint; no changes to the GPU runtime environment.
param([switch]$Fix)

$ErrorActionPreference = 'Stop'
$projectPath = Split-Path -Parent $PSScriptRoot
$environmentPath = Join-Path (Split-Path -Parent $projectPath) 'env'
$uvPath = Join-Path $environmentPath 'tools/bin/uv.exe'
$cachePath = Join-Path $environmentPath 'cache/uv'
if (-not (Test-Path -LiteralPath $uvPath -PathType Leaf)) {
    throw "uv was not found at $uvPath. Use the uv installation documented in README.md."
}

$exitCode = 1
Push-Location -LiteralPath $projectPath
try {
    $formatArguments = if ($Fix) { @('format', '.') } else { @('format', '--check', '.') }
    & $uvPath --cache-dir $cachePath tool run --from ruff==0.14.1 ruff @formatArguments
    $formatExitCode = $LASTEXITCODE
    & $uvPath --cache-dir $cachePath tool run --from ruff==0.14.1 ruff check .
    $lintExitCode = $LASTEXITCODE
    $exitCode = if ($formatExitCode -ne 0) { $formatExitCode } else { $lintExitCode }
} finally {
    Pop-Location
}
exit $exitCode
