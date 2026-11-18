import copy,datetime as dt,json,socket,unittest
from pathlib import Path
from unittest.mock import patch
from decimal import Decimal
from quota_http_fixture import HttpApi
import decimal_quota as d,reconciliation
RAW=json.loads(Path(__file__).with_name('quota-captured-body.json').read_bytes())
class Quota(unittest.TestCase):
 def receipt(self,used,reserved='0s'):
  raw=copy.deepcopy(RAW);raw['gpuQuota']['timeUsed']=used;raw['gpuQuota']['timeReserved']=reserved
  api=HttpApi([raw])
  with patch.object(socket.socket,'connect',side_effect=AssertionError('NETWORK FORBIDDEN')):value=d.read(api)
  self.assertEqual(len(api.sends),1);self.assertEqual(len(api.prior_calls),1);self.assertIs(api.clients[0][0]._response_processor,api.clients[0][1]);self.assertNotIn('parse_error',value)
  return value
 def test_genuine_installed_transport_nanoseconds_and_days(self):
  for text,micros in [('42.306700500s',42306700),('0.000000001s',0),('0.000000999s',0),('0.000001000s',1),('108000.123456789s',108000123456),('77433.978s',77433978000),('1.5s',1500000)]:
   row=self.receipt(text);self.assertEqual(d.verify_projection(row)['used_s'],Decimal(text[:-1]));self.assertEqual(row['typed_projection']['used_s']['total_microseconds'],micros)
 def test_exact_fresh_baseline_and_movement_reserved_nanos_reject(self):
  policy=reconciliation.load_policy();row=self.receipt('2714.079s');reconciliation.validate_quota(row,policy)
  for used,reserved in [('2714.079000001s','0s'),('2714.079s','0.000000001s')]:
   with self.assertRaises(ValueError):reconciliation.validate_quota(self.receipt(used,reserved),policy)
  bad=copy.deepcopy(row);bad['typed_projection']['used_s']['microseconds']+=1
  with self.assertRaises(ValueError):d.validate(bad,policy)
  for key in ('total_allowed_s','minimum_allowed_s'):
   bad=copy.deepcopy(row);bad['wire_seconds'][key]='21600'
   with self.assertRaises(ValueError):d.validate(bad,policy)
  with self.assertRaises(ValueError):d.validate(row,policy,dt.datetime.fromisoformat(row['utc'])+dt.timedelta(seconds=121))
