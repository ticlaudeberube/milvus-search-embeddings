@echo off
if exist .venv\Scripts\activate (
    call .venv\Scripts\activate
) else if exist py312-env\Scripts\activate (
    call py312-env\Scripts\activate
) else (
    echo No virtual environment found
    pause
    exit /b 1
)
cmd /k