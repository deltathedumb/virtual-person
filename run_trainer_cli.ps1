Set-Location $PSScriptRoot
python -m virtual_person.trainer_cli wizard
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "The CLI trainer failed. Install the project first with:"
    Write-Host "    python -m pip install -e ."
    Read-Host "Press Enter to close"
}
