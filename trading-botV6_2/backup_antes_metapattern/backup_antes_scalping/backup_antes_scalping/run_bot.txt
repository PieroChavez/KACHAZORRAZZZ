@echo off
cd /d "%~dp0"
echo Activando entorno virtual...
call venv\Scripts\activate.bat
echo Bot corriendo sin limite de tiempo. Presiona Ctrl+C para detener.
python src/bot.py
pause
