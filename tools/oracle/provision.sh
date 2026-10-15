#!/bin/zsh
# Provision ImagoKit's Python oracle: References/oracle-venv (References is a link to the org folder's
# References/ImagoKit) from tools/oracle/references-lock.txt, wheels only, --no-deps, then `pip check` and the freeze
# into References/oracle-venv-freeze.txt. Run it under the suite's memory cap:
#   ../tools/memcap.sh -- tools/oracle/provision.sh [python3.14]
# Oracles then run as References/oracle-venv/bin/python -I -B <script>, each under ../tools/memcap.sh.
set -euo pipefail
cd "${0:A:h}/../.."
py=${1:-python3}
[[ $($py -c 'import sys; print("%d.%d" % sys.version_info[:2])') == 3.14 ]] || { print -u2 "provision: needs CPython 3.14"; exit 1 }
venv=References/oracle-venv
[[ -d $venv ]] || $py -m venv $venv
$venv/bin/python -I -m pip install --disable-pip-version-check --only-binary=:all: --no-deps \
  -r tools/oracle/references-lock.txt
$venv/bin/python -I -m pip check
$venv/bin/python -I -m pip freeze --all > References/oracle-venv-freeze.txt
$venv/bin/python -I -c 'import sys, numpy, scipy, skimage; print("python", sys.version.split()[0], "numpy", numpy.__version__, "scipy", scipy.__version__, "scikit-image", skimage.__version__)'
