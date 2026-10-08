"""Verify frozen/readback identities and Python portions of inherited notebook cells."""
import hashlib
import ast
import json
from pathlib import Path

p=Path(__file__).resolve().parent
i=json.loads((p/'identity.json').read_text())
for file,key in [('hybrid-runtime.zlib','source_payload_sha256'),('branch-packets.zlib','packets_payload_sha256')]:
 assert hashlib.sha256((p/'dataset-readback'/file).read_bytes()).hexdigest()==i[key]
readback={'passed':True,**{k:i[k] for k in ('source_payload_sha256','packets_payload_sha256')}}
(p/'dataset-readback.json').write_text(json.dumps(readback,indent=2)+'\n')
nb=json.loads((p/'kernel/notebook.ipynb').read_text()); cells=[]; magic=[]
for n,c in enumerate(nb['cells']):
 if c['cell_type']!='code':continue
 source=''.join(c['source'])
 if source.startswith('%%writefile '):
  magic.append({'cell':n,'kind':'inherited file literal'});continue
 lines=[]
 for line in source.splitlines(True):
  if line.lstrip().startswith(('!','%')):
   lines.append(line[:len(line)-len(line.lstrip())]+'pass\n');magic.append({'cell':n,'kind':'inherited shell/IPython statement'})
  else:lines.append(line)
 compile(''.join(lines),f'cell{n}','exec',flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT);cells.append(n)
assert '"self_test": "PASS"' in (p/'qualification-final.log').read_text()
assert 'Ran 8 tests' in (p/'clock-tests-clean.log').read_text() and '\nOK\n' in (p/'clock-tests-clean.log').read_text()
assert 'Ran 2 tests' in (p/'terminal-tests.log').read_text() and '\nOK\n' in (p/'terminal-tests.log').read_text()
(p/'LOCAL-QUALIFICATION.json').write_text(json.dumps({'passed':True,'source_payload_sha256':i['source_payload_sha256'],'packets_payload_sha256':i['packets_payload_sha256'],'branch':'qualification-final.log; four recorded ledgers and maximum cancellation plus two final maxima','worker_checks':{'clock':8,'terminal':2},'test_harness_failure':'Combined run:8 clock passes plus missing port_regressions import; copied helper then terminal2/2 passed','dataset':'downloaded payload hashes matched','compiled_python_portions':cells,'inherited_ipython_exclusions':magic,'risk':'risk-before.json and risk-final.json; bounded diagnostic, added admission branches raise scores; no canonical gate or overall risk improvement claim'},indent=2)+'\n')
print('PASS: freeze/readback and recorded behavioral qualification')
