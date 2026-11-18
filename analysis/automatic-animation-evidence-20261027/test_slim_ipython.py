"""Run inline cells vs slim stubs in a real IPython shell; compare written patch bytes, namespace values and error stop."""
import json,hashlib,sys,shutil,pathlib
from IPython.core.interactiveshell import InteractiveShell
full=json.load(open(sys.argv[1]));slim=json.load(open(sys.argv[2]))
src=lambda nb,i:''.join(nb['cells'][i]['source']) if isinstance(nb['cells'][i]['source'],list) else nb['cells'][i]['source']
sh=InteractiveShell.instance()
# (a) cell 3 inline -> bytes A; slim stub -> bytes B
r=sh.run_cell(src(full,3));assert r.success;A=pathlib.Path('/kaggle/harness-changes.patch').read_bytes();pathlib.Path('/kaggle/harness-changes.patch').unlink()
r=sh.run_cell(src(slim,3));assert r.success,r.error_in_exec;B=pathlib.Path('/kaggle/harness-changes.patch').read_bytes()
print('patch inline',len(A),'slim',len(B),'EQUAL',A==B)
# (b) python cell semantics through nested run_cell: whitespace-only lines inside a triple-quoted string
probe='x = """a\n \nb"""\ny = len(x)\n'
ds=pathlib.Path('/kaggle/input/datasets/pranavagrawai/m2-animation-peak-cell-bodies-20261031');(ds/'probe.txt').write_bytes(probe.encode())
sh.run_cell(probe);inline=(sh.user_ns['x'],sh.user_ns['y'])
del sh.user_ns['x'];sh.run_cell(f"_frozen_result = get_ipython().run_cell(_frozen_cell('probe.txt', {hashlib.sha256(probe.encode()).hexdigest()!r}).decode('utf-8'), store_history=False)\n_frozen_result.raise_error()\n")
print('namespace inline',inline,'slim',(sh.user_ns['x'],sh.user_ns['y']),'EQUAL',inline==(sh.user_ns['x'],sh.user_ns['y']))
# (c) failure inside the frozen body must fail the outer cell
bad=b'raise RuntimeError("boom")\n';(ds/'bad.txt').write_bytes(bad)
r=sh.run_cell(f"_frozen_result = get_ipython().run_cell(_frozen_cell('bad.txt', {hashlib.sha256(bad).hexdigest()!r}).decode('utf-8'), store_history=False)\n_frozen_result.raise_error()\nprint('SHOULD NOT PRINT')\n")
print('outer cell failed on inner error:',not r.success)
# (d) tampered body is refused
(ds/'cell06.txt').write_bytes(b'print(1)\n');r=sh.run_cell(src(slim,6));print('tampered body refused:',not r.success and 'hash mismatch' in repr(r.error_in_exec))
