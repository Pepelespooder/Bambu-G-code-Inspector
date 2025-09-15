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
  echo When run via this .BAT, an interactive hotkey menu is available:
  echo   L: Layer metrics   F: Flow   E: E^/mm   K: Corner stress   O: Cooling
  echo.
  pause
  exit /b 1
)

rem Use interactive prompt in Python so we can offer 'P to save PNG'
%PYTHON% "%SCRIPT_DIR%gcode_inspector.py" --interactive %*
