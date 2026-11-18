"""Exclusive bounded raw artifacts; IPC carries references, never large bodies."""
import hashlib
import json
import os
from pathlib import Path
import stat
import time
import uuid

MAX_RAW_BYTES=8*1024**2


def version(value):return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)


def new_file(root,label):
    root=Path(root);root.mkdir(exist_ok=True,parents=True)
    path=root/(label+'-'+uuid.uuid4().hex+'.raw')
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    return path,os.fdopen(fd,'wb',buffering=0)


def reference(path,root,complete,started,ended):
    digest=hashlib.sha256();size=0
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(65536),b''):digest.update(block);size+=len(block)
    return dict(path=str(Path(path).relative_to(root)),sha256=digest.hexdigest(),bytes=size,
                complete=complete,capture_start=started,capture_end=ended)


def save(root,label,data,started=None):
    if len(data)>MAX_RAW_BYTES:raise ValueError('raw artifact size cap')
    path,stream=new_file(root,label)
    start=time.monotonic() if started is None else started
    with stream:stream.write(data)
    return reference(path,Path(root),True,start,time.monotonic())


def verify(root,receipt,end):
    root=Path(root).resolve();path=root/receipt['path']
    if path.is_symlink() or path.resolve().parent!=root:raise ValueError('artifact path escape/symlink')
    before=path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size>MAX_RAW_BYTES:raise ValueError('raw artifact type/size')
    digest=hashlib.sha256();size=0
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(65536),b''):
            if time.monotonic()>=end:raise TimeoutError('artifact verify deadline')
            digest.update(block);size+=len(block)
    after=path.stat()
    if version(before)!=version(after) or size!=receipt['bytes'] or digest.hexdigest()!=receipt['sha256']:
        raise ValueError('artifact drift')
    return receipt['complete']
