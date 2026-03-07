@echo off
REM =============================================================================
REM setup_venv.bat - Create universal venv and install all requirements 
REM =============================================================================

set SCRIPT_DIR=%~dp0
set VENV_DIR=%SCRIPT_DIR%.venv

echo Creating Universal Python venv at %VENV_DIR% ...
python -m venv "%VENV_DIR%"

echo Activating venv and installing requirements...
call "%VENV_DIR%\Scripts\activate.bat"

python -m pip install --upgrade pip setuptools wheel

echo Installing Depth Sensing requirements...
pip install -r "%SCRIPT_DIR%Depth sensing\requirements.txt"

echo Installing Edge Pipeline requirements...
pip install -r "%SCRIPT_DIR%edge_pipeline\requirements.txt"

echo.
echo Done. Activate with:  .venv\Scripts\activate
pause
