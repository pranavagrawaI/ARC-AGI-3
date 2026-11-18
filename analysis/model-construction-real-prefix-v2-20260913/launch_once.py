#!/usr/bin/env python3
"""Authorized single launch. Never render credentials or other environment values."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
if (HERE / 'STARTED.json').exists() or (HERE / 'raw').exists():
    raise SystemExit('Experiment already started; no retry allowed')
if json.loads((HERE / 'PRELAUNCH.json').read_text()).get('passed') is not True:
    raise SystemExit('Prelaunch did not pass')
key = None
with Path('/home/pranav/projects/arc-agi-3/.env').open() as stream:
    for line in stream:
        name, separator, value = line.strip().removeprefix('export ').partition('=')
        if separator and name.strip() == 'OPENROUTER_API_KEY':
            if key is not None:
                raise SystemExit('Duplicate authorized key entry')
            value = value.strip()
            key = ast.literal_eval(value) if value.startswith(('"', "'")) else value
if not isinstance(key, str) or not key:
    raise SystemExit('Authorized key unavailable')
# Same single-launch guard and key parser; no secret-bearing subprocess.
import run
run.AUTHORIZED_KEY = key
sys.argv = [str(HERE / 'run.py'), '--execute']
run.main()
