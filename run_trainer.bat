@echo off
setlocal
cd /d "%~dp0"
python -m virtual_person.trainer_ui
if errorlevel 1 (
    echo.
    echo Trainer failed to launch. Install the project first with:
    echo     python -m pip install -e .
    pause
)
