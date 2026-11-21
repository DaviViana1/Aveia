@echo off
setlocal EnableExtensions
chcp 65001 >nul
set "PYTHONUTF8=1"
cd /d "%~dp0"
title Aveia
echo.
echo  Aveia: notas por voz que rodam no seu computador
echo.

rem 1. Python 3.11 ou mais novo
call :achar_python
if not defined PY goto :sem_python

rem 2. O computador aguenta? (leitura rápida: Windows, memória, disco, processador)
echo  Conferindo este computador...
echo.
%PY% transcritor\perfil_pc.py
if errorlevel 2 (
    echo.
    set /p RESP= Continuar mesmo assim? Pode não funcionar. ^(S/N^) 
    if /i not "%RESP%"=="S" exit /b 1
)
echo.

rem 3. Ambiente e bibliotecas (só na primeira vez, ou quando requirements.txt mudar)
if not exist ".venv\Scripts\python.exe" (
    echo  Criando o ambiente do Aveia...
    %PY% -m venv .venv
    if errorlevel 1 goto :erro
)
fc requirements.txt ".venv\requisitos.txt" >nul 2>&1
if errorlevel 1 (
    echo  Instalando as bibliotecas ^(só na primeira vez; pode levar alguns minutos^)...
    ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
    if errorlevel 1 goto :erro
    copy /y requirements.txt ".venv\requisitos.txt" >nul
)

rem 4. Aveia
echo.
echo  Abrindo o Aveia no navegador. Na primeira vez ele baixa o modelo de voz ^(cerca de 1,6 GB^).
echo  Para encerrar, feche esta janela ou use o parar.bat.
echo.
".venv\Scripts\python.exe" app.py
if errorlevel 1 goto :erro
exit /b 0

:achar_python
set "PY="
py -3 -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, 11) else 1)" >nul 2>&1 && set "PY=py -3"
if defined PY goto :eof
python -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, 11) else 1)" >nul 2>&1 && set "PY=python"
goto :eof

:sem_python
echo  O Aveia precisa do Python 3.11 ou mais novo, e não encontrei no seu computador.
echo.
where winget >nul 2>&1
if errorlevel 1 goto :python_manual
echo  Posso instalar o Python 3.12 pelo winget, só para o seu usuário ^(sem administrador^).
echo  Ao responder S, você aceita os termos do winget e do Python.
set /p RESP= Instalar agora? ^(S/N^) 
if /i not "%RESP%"=="S" goto :python_manual
winget install --id Python.Python.3.12 -e --scope user --accept-package-agreements --accept-source-agreements
echo.
echo  Pronto. Feche esta janela e abra o iniciar.bat de novo.
pause
exit /b 0

:python_manual
echo.
echo  Baixe o Python em https://www.python.org/downloads/ e, no instalador, marque
echo  "Add python.exe to PATH". Depois abra o iniciar.bat de novo.
start "" "https://www.python.org/downloads/"
pause
exit /b 1

:erro
echo.
echo  Algo deu errado. Confira a internet e o espaço livre em disco e tente de novo.
echo  Se continuar, abra um problema em https://github.com/DaviViana1/Aveia/issues
pause
exit /b 1
