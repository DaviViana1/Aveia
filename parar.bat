@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
rem Encerra só o Aveia desta pasta (o programa que usa o .venv daqui e escuta na porta do Aveia). Nada mais é tocado.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ini = if ($env:AVEIA_PORTA) { [int]$env:AVEIA_PORTA } else { 8765 }; $fim = if ($env:AVEIA_PORTA) { $ini } else { $ini + 34 }; $meu = (Join-Path (Get-Location) '.venv\Scripts\python.exe').ToLower(); $achou = $false; Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.LocalPort -ge $ini -and $_.LocalPort -le $fim } | ForEach-Object { $c = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $_.OwningProcess) -ErrorAction SilentlyContinue; $pai = if ($c) { Get-CimInstance Win32_Process -Filter ('ProcessId=' + $c.ParentProcessId) -ErrorAction SilentlyContinue }; $meusTextos = @($c.ExecutablePath, $pai.ExecutablePath) | Where-Object { $_ -and $_.ToLower() -eq $meu }; if ($meusTextos) { if ($pai) { Stop-Process -Id $pai.ProcessId -Force -ErrorAction SilentlyContinue }; Stop-Process -Id $c.ProcessId -Force; $achou = $true } }; if ($achou) { 'Aveia encerrado.' } else { 'O Aveia desta pasta não está rodando.' }"
timeout /t 3 >nul
