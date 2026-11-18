#!/usr/bin/env python3
import argparse,hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--authorize-one-run',required=True);args=p.parse_args()
auth=HERE/'AUTHORIZATION.json'
if not auth.exists(): raise SystemExit('LAUNCH_DISABLED: separate one-run authorization is absent')
data=json.loads(auth.read_text())
if data.get('used') is not False or hashlib.sha256(args.authorize_one_run.encode()).hexdigest()!=data.get('token_sha256'): raise SystemExit('LAUNCH_DISABLED: invalid or consumed one-run authorization')
raise SystemExit('LAUNCH_DISABLED_PLACEHOLDER: coordinator must replace this terminal guard only after separate authorization; no push command is present')
