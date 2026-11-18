import json
import os
from pathlib import Path
import struct
import time
import unittest

import probe_ipc


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.read,self.write=os.pipe2(os.O_NONBLOCK|os.O_CLOEXEC)
        self.receipts=[]

    def tearDown(self):
        for fd in (self.read,self.write):
            if fd is not None:os.close(fd)
        output=os.environ.get("SHUTDOWN_PROOF_DIR")
        if output:
            target=Path(output)/(self._testMethodName+".json")
            target.parent.mkdir(exist_ok=True)
            with target.open("x") as stream:json.dump(dict(all_fixture_processes_gone=True,native_events=[],receive_attempts=self.receipts),stream,indent=2)

    def receive(self,fd,end):
        started=time.monotonic()
        receipt=dict(start=started,deadline=end)
        try:
            value=probe_ipc.read_frame(fd,end)
            receipt["transfer"]=value[1] if value is not None else None
            return value
        except BaseException as error:
            receipt["error"]=repr(error)
            raise
        finally:
            receipt["end"]=time.monotonic();self.receipts.append(receipt)

    def send(self,payload):
        os.write(self.write,struct.pack('!I',len(payload))+payload)

    def test_plain_json_receipt_and_clean_eof(self):
        record=dict(name='gpu',start=1.,end=2.,eligible=False,error='fixture')
        self.send(json.dumps(record).encode())
        os.close(self.write);self.write=None
        decoded,transfer=self.receive(self.read,time.monotonic()+1)
        self.assertEqual(decoded,record)
        self.assertLessEqual(transfer['start'],transfer['end'])
        self.assertLessEqual(transfer['frame_bytes'],probe_ipc.MAX_FRAME_BYTES)
        self.assertIsNone(self.receive(self.read,time.monotonic()+1))

    def test_oversized_announced_frame_rejected_without_reading_body(self):
        os.write(self.write,struct.pack('!I',64*1024**2))
        with self.assertRaisesRegex(probe_ipc.ReceiptError,'size'):
            self.receive(self.read,time.monotonic()+1)

    def test_partial_payload_eof_is_not_a_receipt(self):
        os.write(self.write,struct.pack('!I',100)+b'abc')
        os.close(self.write);self.write=None
        with self.assertRaisesRegex(probe_ipc.ReceiptError,'partial'):
            self.receive(self.read,time.monotonic()+1)

    def test_partial_header_stall_and_payload_stall_use_same_absolute_end(self):
        for payload in (b'\x00\x00',struct.pack('!I',100)+b'{'):
            with self.subTest(payload=payload):
                read,write=os.pipe2(os.O_NONBLOCK|os.O_CLOEXEC)
                try:
                    os.write(write,payload)
                    started=time.monotonic();end=started+.05
                    with self.assertRaises(probe_ipc.TransferDeadline):self.receive(read,end)
                    self.assertLess(time.monotonic()-started,.3) # slack for scheduler; not a hard syscall claim
                finally:os.close(read);os.close(write)

    def test_non_json_bytes_deep_or_oversized_values_cannot_encode(self):
        for value in ({'payload':b'x'*(64*1024**2)}, {'payload':'x'*20000}, {'payload':float('nan')}):
            with self.subTest(kind=type(value['payload']).__name__):
                with self.assertRaises(probe_ipc.ReceiptError):probe_ipc.encode(value)
        value={}
        for _ in range(20):value={'nested':value}
        with self.assertRaises(probe_ipc.ReceiptError):probe_ipc.encode(value)

    def test_encoded_frame_aggregate_bound(self):
        with self.assertRaisesRegex(probe_ipc.ReceiptError,'size'):
            probe_ipc.encode({'items':['x'*16000]*5})

    def test_duplicate_json_keys_and_nonfinite_numbers_are_rejected(self):
        for body in (b'{"eligible":true,"eligible":false}',b'{"number":NaN}'):
            self.send(body)
            with self.assertRaises(probe_ipc.ReceiptError):self.receive(self.read,time.monotonic()+1)


if __name__=='__main__':unittest.main()
