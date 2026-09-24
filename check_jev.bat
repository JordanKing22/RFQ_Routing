@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Jev API check

if not exist "server.py" (
  echo.
  echo  server.py is missing from this folder. If you opened this file from inside a zip,
  echo  extract the whole folder first, then run it from the extracted folder.
  echo.
  pause
  exit /b 1
)

set "PY="
call :try_python "py -3"
if not defined PY call :try_python "python"
if not defined PY call :try_python "python3"
if not defined PY (
  echo.
  echo  Python 3.10 or newer was not found. Install it from https://www.python.org/downloads/
  goto :end
)
%PY% check_jev.py %*
goto :end

:try_python
%~1 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 set "PY=%~1"
exit /b 0

:end
echo.
pause
endlocal
