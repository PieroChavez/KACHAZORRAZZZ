@echo off
cd /d "%~dp0"
echo [%date% %time%] Retraining ML regime models...
call venv\Scripts\python scripts\retrain_models.py
echo.
echo Done. Press any key to exit.
pause >nul
