"""No retries or hidden token exposure when usage is malformed."""
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch
from continuation import Transport, Stop


class Reply:
    status_code=200
    def __init__(self,usage):self.usage=usage
    def json(self):return {'usage':{'completion_tokens':self.usage}}


class TransportTests(unittest.TestCase):
    def tearDown(self):signal.setitimer(signal.ITIMER_REAL,0)

    def test_valid_usage_is_receipted_and_boolean_usage_stops_unknown(self):
        for usage,valid in [(3,True),(True,False)]:
            with tempfile.TemporaryDirectory() as directory:
                out=Path(directory);transport=Transport(out,'http://localhost/v1',9)
                with patch('worker.REAL_POST',return_value=Reply(usage)) as post:
                    if valid:
                        transport.post('http://localhost/v1/chat/completions',json={'max_tokens':7})
                        self.assertEqual(transport.tokens,3)
                        self.assertIsNone(transport.unresolved)
                    else:
                        with self.assertRaises(Stop) as caught:
                            transport.post('http://localhost/v1/chat/completions',json={'max_tokens':7})
                        self.assertEqual(caught.exception.status,'invalid_usage')
                        self.assertEqual(transport.tokens,0)
                        self.assertEqual(transport.unresolved,{'call':1,'reserved_tokens':7})
                    self.assertEqual(post.call_count,1)
                    rows=[json.loads(line) for line in (out/'wire.jsonl').read_text().splitlines()]
                    self.assertEqual(rows[0]['body']['seed'],9)
                    if not valid:self.assertEqual(rows[-1]['event'],'failure')


if __name__=='__main__':unittest.main()
