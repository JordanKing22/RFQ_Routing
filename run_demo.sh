#!/usr/bin/env bash
# RFQ Router demo (Jev) for macOS and Linux. Same steps as run_demo.bat.
#   ./run_demo.sh                check the key on the first run, then start and open the browser
#   ./run_demo.sh --port 9000    options pass through to server.py
set -u
cd "$(dirname "$0")" || exit 1

if [ ! -f server.py ]; then
  echo "server.py is missing from this folder. Extract the whole folder first, then run it from there." >&2
  exit 1
fi

# ---- find Python 3.10+ (no packages needed, standard library only) ----
PY=""
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 &&
     "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
    PY="$candidate"
    break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.10 or newer was not found. Install it from https://www.python.org/downloads/" >&2
  exit 1
fi

# ---- first run only: prove the Jev key works with one real call ----
if [ ! -f cache/.jev_ok ]; then
  echo
  echo " First run: checking your Jev API key with one sample email..."
  if ! "$PY" check_jev.py; then
    echo
    echo " The Jev check did not pass. See the message above."
    echo " You can still open the demo; it will show the same error when you route."
    if [ -t 0 ]; then
      answer=""
      printf '  Open the demo anyway? [Y/N] '
      read -r answer
      case "$answer" in
        [Yy]*) ;;
        *) exit 1 ;;
      esac
    fi
  fi
fi

echo
exec "$PY" server.py "$@"
