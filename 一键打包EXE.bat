@echo off
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10 or 3.11 from https://www.python.org/downloads/windows/
    echo Make sure to check 'Add Python to PATH' during installation.
    pause
    exit /b 1
)
python build_windows_exe.py
pause
