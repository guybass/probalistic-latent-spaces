$ErrorActionPreference = "Stop"

$projectDir = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$repositoryPython = Join-Path $PSScriptRoot "..\..\..\.venv\Scripts\python.exe"

if (Test-Path -LiteralPath $repositoryPython) {
    $python = (Resolve-Path -LiteralPath $repositoryPython).Path
} else {
    $python = (Get-Command python -ErrorAction Stop).Source
}

& $python (Join-Path $PSScriptRoot "paper_harness.py") run-reproduce `
    --project-dir $projectDir @args
exit $LASTEXITCODE
