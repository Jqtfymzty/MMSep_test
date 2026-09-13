@echo off
chcp 65001 >nul
setlocal
set "PYTHONIOENCODING=utf-8"

pushd "%~dp0"
if errorlevel 1 (
    echo [ERROR] Cannot enter the project directory.
    exit /b 1
)

if exist ".venv\Scripts\python.exe" (
    set "PROJECT_PYTHON=.venv\Scripts\python.exe"
) else (
    set "PROJECT_PYTHON=python"
)

echo [INFO] Python executable: %PROJECT_PYTHON%
"%PROJECT_PYTHON%" -m pytest tests -v
set "TEST_EXIT_CODE=%ERRORLEVEL%"

if not "%TEST_EXIT_CODE%"=="0" (
    echo [ERROR] Pytest exit code: %TEST_EXIT_CODE%
) else (
    echo [INFO] Test run completed.
)

popd
exit /b %TEST_EXIT_CODE%
