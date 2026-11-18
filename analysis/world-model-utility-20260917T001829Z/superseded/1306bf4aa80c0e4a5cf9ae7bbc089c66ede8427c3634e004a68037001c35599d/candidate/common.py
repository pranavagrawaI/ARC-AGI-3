import hashlib,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
TASK='WORLD-MODEL-UTILITY'
PREFIX=['RIGHT','RIGHT','DOWN','DOWN','LEFT','LEFT','UP','UP']
MODEL='qwen/qwen3.8-27b'
CAPS={'constructor':{'requests':6,'completion_per_request':4096,'completion':24576,'prompt':600000,'soft':240,'hard':245,'allocation':'0.1910688'},
      'control':{'requests':120,'completion_per_request':1024,'completion':122880,'prompt':1500000,'soft':1800,'hard':1805,'allocation':'0.634344'},
      'treatment':{'requests':120,'completion_per_request':1024,'completion':122880,'prompt':1500000,'soft':1800,'hard':1805,'allocation':'0.634344'}}
def encoded(v):return (json.dumps(v,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n').encode()
def digest(raw):return hashlib.sha256(raw).hexdigest()
def put(path,value):
 path=Path(path);raw=encoded(value)
 with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return digest(raw)
def read(path):return json.loads(Path(path).read_bytes())
def journal(path,value):
 with Path(path).open('ab') as f:f.write(encoded(value));f.flush();os.fsync(f.fileno())
def hash_file(path):return digest(Path(path).read_bytes())
