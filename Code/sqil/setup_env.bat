@echo off
setlocal EnableDelayedExpansion

echo =====================================
echo Configurando entorno SQIL + MuJoCo 🧠
echo =====================================

REM Crear entorno virtual
if not exist ".venv" (
    echo 🔧 Creando entorno virtual...
    python -m venv .venv
) else (
    echo ✅ Entorno virtual ya existe
)

REM Activar entorno virtual
call .venv\Scripts\activate

REM Instalar requirements
echo 📦 Instalando dependencias desde requirements.txt...
pip install --upgrade pip
pip install -r requirements.txt

REM Verificar MuJoCo 2.1.0
set MUJOCO_DIR=%USERPROFILE%\.mujoco\mujoco210
if exist "%MUJOCO_DIR%" (
    echo ✅ MuJoCo 2.1.0 ya está instalado en: %MUJOCO_DIR%
) else (
    echo ❌ No se encontró MuJoCo 2.1.0 en: %MUJOCO_DIR%
    echo.
    echo 🧾 Por favor, sigue estos pasos:
    echo 1. Descarga MuJoCo 2.1.0 desde:
    echo    https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-windows-x86_64.zip
    echo 2. Extrae el ZIP en:
    echo    %USERPROFILE%\.mujoco\mujoco210
    echo 3. Verifica que el archivo bin\mujoco210.dll exista.
    echo.
    echo ⚠️  El entorno funcionará solo si esa carpeta está correctamente configurada.
    pause
)

echo.
echo 🚀 Para ejecutar tu script, usa:
echo call .venv\Scripts\activate && python sqil_2.py

REM Agregar MuJoCo bin temporalmente al PATH
set PATH=%USERPROFILE%\.mujoco\mujoco210\bin;%PATH%

pause
