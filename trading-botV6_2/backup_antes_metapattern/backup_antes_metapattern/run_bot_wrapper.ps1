$host.UI.RawUI.WindowTitle = "SMC Trading Bot"
Write-Host "========================================"
Write-Host "  SMC Trading Bot - Auto Restart Wrapper"
Write-Host "========================================"
Write-Host "  Ctrl+C  ->  Detiene permanentemente"
Write-Host "  Crash   ->  Reinicio automatico en 5s"
Write-Host "========================================"
Write-Host ""

do {
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Iniciando bot..."
    Write-Host ""

    $proc = Start-Process -FilePath ".\venv\Scripts\python.exe" -ArgumentList "src\bot.py" -NoNewWindow -PassThru -Wait

    $exitCode = $proc.ExitCode

    Write-Host ""
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Bot finalizado con codigo: $exitCode"

    if ($exitCode -eq 0) {
        Write-Host ""
        Write-Host "Cierre manual detectado (Ctrl+C). Saliendo."
        Write-Host "Presione Enter para cerrar..."
        $null = Read-Host
        break
    } else {
        Write-Host ""
        Write-Host "CRASH detectado (codigo $exitCode) - Reiniciando en 5 segundos..."
        Start-Sleep -Seconds 5
    }
} while ($true)
