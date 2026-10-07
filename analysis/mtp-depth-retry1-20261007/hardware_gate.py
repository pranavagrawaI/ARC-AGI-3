"""First notebook cell: retain actual inventory and reject the wrong accelerator."""
import json
import subprocess
from pathlib import Path

hardware_dir = Path('/kaggle/working/mtp')
hardware_dir.mkdir(parents=True, exist_ok=True)
inventory = subprocess.run(
    ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits'],
    capture_output=True, text=True, timeout=15)
rows = [line.strip().split(',') for line in inventory.stdout.splitlines() if line.strip()]
valid = (inventory.returncode == 0 and len(rows) == 1 and len(rows[0]) == 3
         and 'RTX PRO 6000' in rows[0][0]
         and rows[0][1].strip().isdigit() and int(rows[0][1]) >= 90000)
receipt = {'run_id': 'mtp-depth-retry1-20261007', 'returncode': inventory.returncode,
           'stdout': inventory.stdout, 'stderr': inventory.stderr, 'valid': valid}
(hardware_dir / 'hardware.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('retry hardware:', json.dumps(receipt), flush=True)
if not valid:
    raise RuntimeError('Retry requires exactly one RTX PRO 6000 with >=90000 MiB; abort before loading')
