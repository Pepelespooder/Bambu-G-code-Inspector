@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "PYTHON=py -3"

rem Prefer py launcher if available; fallback to python
where py >nul 2>nul
if errorlevel 1 set "PYTHON=python"

if "%~1"=="" (
  echo Drag-and-drop a .gcode file onto this file, or run:
  echo   %PYTHON% "%SCRIPT_DIR%gcode_inspector.py" "path\to\file.gcode"
  echo.
  pause
  exit /b 1
)

%PYTHON% "%SCRIPT_DIR%gcode_inspector.py" %*
echo.
pause

