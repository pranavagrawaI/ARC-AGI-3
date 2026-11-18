"""Bounded JSON receipt framing, Linux nonblocking pipes; no pickle."""
import json
import math
import os
import select
import struct
import time

MAX_FRAME_BYTES=65536
MAX_STRING_CHARS=16384
MAX_NODES=2048
MAX_DEPTH=12


class ReceiptError(ValueError):
    pass


class TransferDeadline(TimeoutError):
    pass


def validate_tree(value):
    nodes=0
    encoded_size=0
    def spend(count):
        nonlocal encoded_size
        encoded_size+=count
        if encoded_size>MAX_FRAME_BYTES:raise ReceiptError('receipt encoded size limit')
    def visit(item,depth):
        nonlocal nodes
        nodes+=1
        if nodes>MAX_NODES or depth>MAX_DEPTH:raise ReceiptError('receipt shape limit')
        scalar=(item is None or type(item) is bool or type(item) is int and item.bit_length()<=128
                or type(item) is float and math.isfinite(item) or type(item) is str and len(item)<=MAX_STRING_CHARS)
        if scalar:
            spend(len(json.dumps(item,ensure_ascii=True,allow_nan=False)))
            return
        if type(item) is list:
            spend(2+max(0,len(item)-1))
            for child in item:visit(child,depth+1)
            return
        if type(item) is dict:
            spend(2+max(0,len(item)-1))
            for key,child in item.items():
                if type(key) is not str or len(key)>256:raise ReceiptError('receipt key limit')
                spend(len(json.dumps(key,ensure_ascii=True))+1)
                visit(child,depth+1)
            return
        raise ReceiptError('receipt requires bounded JSON types/strings/numbers')
    visit(value,0)


def encode(value):
    validate_tree(value)
    payload=json.dumps(value,ensure_ascii=True,separators=(',',':'),allow_nan=False).encode('ascii')
    if not 0<len(payload)<=MAX_FRAME_BYTES:raise ReceiptError('receipt encoded size limit')
    return payload


def before_end(end):
    if not math.isfinite(end) or time.monotonic()>=end:raise TransferDeadline('receipt absolute transfer deadline')


def write_frame(fd,value,end):
    if os.get_blocking(fd):raise ReceiptError('nonblocking receipt descriptor required')
    frame=encode(value)
    pending=memoryview(struct.pack('!I',len(frame))+frame)
    while pending:
        before_end(end)
        try:written=os.write(fd,pending)
        except BlockingIOError:
            select.select([],[fd],[],max(0,end-time.monotonic()))
            continue
        if written<=0:raise ReceiptError('receipt write made no progress')
        pending=pending[written:]
    before_end(end)


def read_exact(fd,size,end,allow_eof=False):
    data=bytearray()
    while len(data)<size:
        before_end(end)
        ready,_,_=select.select([fd],[],[],max(0,end-time.monotonic()))
        before_end(end)
        if not ready:continue
        try:block=os.read(fd,min(8192,size-len(data)))
        except BlockingIOError:continue
        before_end(end)
        if not block:
            if allow_eof and not data:return None
            raise ReceiptError('partial receipt at EOF')
        data.extend(block)
    return bytes(data)


def unique_object(pairs):
    value={}
    for key,item in pairs:
        if key in value:raise ReceiptError('duplicate receipt key')
        value[key]=item
    return value


def reject_constant(value):
    raise ReceiptError('nonfinite receipt number')


def read_frame(fd,end):
    started=time.monotonic()
    if os.get_blocking(fd):raise ReceiptError('nonblocking receipt descriptor required')
    header=read_exact(fd,4,end,allow_eof=True)
    if header is None:return None
    size=struct.unpack('!I',header)[0]
    if not 0<size<=MAX_FRAME_BYTES:raise ReceiptError('receipt announced size limit')
    payload=read_exact(fd,size,end)
    decode_started=time.monotonic()
    try:
        value=json.loads(payload,object_pairs_hook=unique_object,parse_constant=reject_constant)
        validate_tree(value)
    except (ValueError,TypeError,RecursionError,UnicodeError) as error:
        raise ReceiptError('invalid bounded JSON receipt: '+str(error)[:256]) from error
    before_end(end)
    return value,dict(start=started,end=time.monotonic(),decode_start=decode_started,frame_bytes=size)
