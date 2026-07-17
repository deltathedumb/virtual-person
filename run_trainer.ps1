Set-Location $PSScriptRoot
python -m virtual_person.trainer_ui
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Trainer failed to launch. Install the project first with:"
    Write-Host "    python -m pip install -e ."
    Read-Host "Press Enter to close"
}
