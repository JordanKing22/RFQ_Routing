@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title RFQ Router demo (Jev)

if not exist "server.py" (
  echo.
  echo  server.py is missing from this folder. If you opened this file from inside a zip,
  echo  extract the whole folder first, then run it from the extracted folder.
  echo.
  pause
  exit /b 1
)

rem ---- find Python 3.10+ (no packages needed, standard library only) ----
set "PY="
call :try_python "py -3"
if not defined PY call :try_python "python"
if not defined PY call :try_python "python3"
if not defined PY goto :no_python

rem ---- first run only: prove the Jev key works with one real call ----
if exist "cache\.jev_ok" goto :start
echo.
echo  First run: checking your Jev API key with one sample email...
%PY% check_jev.py
if errorlevel 1 (
  echo.
  echo  The Jev check did not pass. See the message above.
  echo  You can still open the demo; it will show the same error when you route.
  choice /c YN /n /m "  Open the demo anyway? [Y/N] "
  if errorlevel 2 goto :end
)

:start
echo.
%PY% server.py %*
goto :end

:no_python
echo.
echo  Python 3.10 or newer was not found.
echo  Install it from https://www.python.org/downloads/ and tick
echo  "Add python.exe to PATH" during setup, then run this file again.
echo.
goto :end

:try_python
%~1 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 set "PY=%~1"
exit /b 0

:end
echo.
pause
endlocal
