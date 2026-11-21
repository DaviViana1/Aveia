@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".git" (
    echo  Esta pasta veio de um ZIP, então não dá para atualizar sozinha.
    echo  Baixe o ZIP novo em https://github.com/DaviViana1/Aveia e extraia por cima
    echo  ^(suas notas ficam no navegador e não são tocadas^).
    pause
    exit /b 0
)
where git >nul 2>&1
if errorlevel 1 (
    echo  O Git não está instalado. Baixe o ZIP novo em https://github.com/DaviViana1/Aveia
    pause
    exit /b 1
)
git pull --ff-only
if errorlevel 1 (
    echo.
    echo  Não consegui atualizar sem sobrescrever mudanças suas nesta pasta. Nada foi alterado.
    pause
    exit /b 1
)
echo.
echo  Atualizado. As bibliotecas novas, se houver, são instaladas na próxima vez que abrir o iniciar.bat.
pause
