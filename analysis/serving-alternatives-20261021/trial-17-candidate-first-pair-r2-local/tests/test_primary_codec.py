import base64,hashlib,json,unittest,zlib
from pathlib import Path
import primary_bundle as p
REAL=Path(__file__).resolve().parents[2]/'trial-16-candidate-s10-r4-operational-local/remote-readback/current-primary-fallback-local/primary-current/PRIMARY.json'
class Codec(unittest.TestCase):
 def test_retained_primary_roundtrip(self):
  original=json.loads(REAL.read_bytes()); original['schema']='CANDIDATE_FIRST_PAIR17_PRIMARY_V1'
  encoded=p.encode(original)
  self.assertLess(len(encoded),16*1024**2)
  self.assertEqual(p.verify(encoded),original)
 def test_corruption_trailing_and_bomb_rejected(self):
  value={'schema':'CANDIDATE_FIRST_PAIR17_PRIMARY_V1','files':{},'capacity_or_rate_qualification':False}
  encoded=p.encode(value); envelope=json.loads(encoded)
  envelope['decoded_sha256']='0'*64
  with self.assertRaises(ValueError):p.verify(json.dumps(envelope).encode())
  for data in (zlib.compress(b'x'*(p.MAX_DECODED+1)), zlib.compress(json.dumps(value).encode())+b'trailing', zlib.compress(json.dumps(value).encode())[:-1]):
   envelope=json.loads(encoded);envelope['compressed_base64']=base64.b64encode(data).decode();envelope['compressed_bytes']=len(data);envelope['compressed_sha256']=hashlib.sha256(data).hexdigest()
   with self.assertRaises(ValueError):p.verify(json.dumps(envelope).encode())
