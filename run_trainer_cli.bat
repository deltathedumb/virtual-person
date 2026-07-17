@echo off
setlocal
cd /d "%~dp0"
python -m virtual_person.trainer_cli wizard
if errorlevel 1 (
    echo.
    echo The CLI trainer failed. Install the project first with:
    echo     python -m pip install -e .
    pause
)
