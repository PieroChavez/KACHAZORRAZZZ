@echo off
cd /d "%~dp0"
title SMC Trading Bot
echo ========================================
echo   SMC Trading Bot - Auto Restart Wrapper
echo ========================================
echo   Ctrl+C para detener permanentemente
echo ========================================
echo.

:restart
echo [%date% %time%] Iniciando bot...
echo.

.\venv\Scripts\python.exe src\bot.py

set EXIT_CODE=%ERRORLEVEL%

echo.
echo [%date% %time%] Bot finalizado con codigo: %EXIT_CODE%

if "%EXIT_CODE%"=="0" (
    echo.
    echo Cierre manual detectado (Ctrl+C). Saliendo.
    echo Presione cualquier tecla para cerrar.
    pause >nul
    exit /b 0
) else (
    echo.
    echo Crash detectado (codigo %EXIT_CODE%) - Reiniciando en 5 segundos...
    timeout /t 5 /nobreak >nul
    goto restart
)
