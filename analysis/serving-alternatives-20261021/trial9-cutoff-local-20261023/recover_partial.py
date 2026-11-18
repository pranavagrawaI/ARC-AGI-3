"""CPU-only local-header recovery, NOT a complete ZIP/central-directory audit.

Keep only complete, descriptor/CRC/size-verified local members. Never resync by
searching for a signature after a failure; stop at the first incomplete member.
Never execute recovered files. Original archive is read-only and hash-bound.
"""
import hashlib,json,struct,time,zlib
from pathlib import Path,PurePosixPath

HERE=Path(__file__).resolve().parent
ARCHIVE=HERE.parent/'trial-9-readiness-local/remote-readback/output-v1-kernel140375737.zip'
EXPECTED='85dc64a5235f47c5c557ee309ab65ee90f033667bc4c84a4ec5bd582d0f6747f'

def recover(destination):
    started=time.monotonic();raw=ARCHIVE.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==EXPECTED
    destination=Path(destination);destination.mkdir(exist_ok=False)
    entries=[];position=0;total=0;seen=set();stop=None
    while position<len(raw):
        start=position
        try:
            if time.monotonic()-started>30:raise ValueError('30s CPU recovery budget reached')
            if len(raw)-position<30:raise ValueError('incomplete local header')
            signature,version,flags,method,mtime,mdate,crc,csize,usize,nlen,xlen=struct.unpack_from('<4s5H3I2H',raw,position)
            if signature!=b'PK\x03\x04':raise ValueError('not a local header; no resynchronization')
            if flags&1 or method not in (0,8) or version>20:raise ValueError('unsupported encrypted/method/version')
            position+=30
            if len(raw)-position<nlen+xlen:raise ValueError('incomplete filename/extra')
            name=raw[position:position+nlen].decode('utf-8' if flags&2048 else 'cp437');position+=nlen+xlen
            path=PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name or name in seen:raise ValueError('unsafe/duplicate member')
            seen.add(name);data_start=position
            if method==8:
                inflater=zlib.decompressobj(-15);parts=[];count=0
                while not inflater.eof:
                    if position==len(raw):raise ValueError('incomplete compressed member')
                    block=raw[position:position+65536];position+=len(block)
                    decoded=inflater.decompress(block,32*1024**2-count+1);count+=len(decoded);parts.append(decoded)
                    if count>32*1024**2 or inflater.unconsumed_tail:raise ValueError('32MiB member limit')
                    if inflater.eof:position-=len(inflater.unused_data)
                value=b''.join(parts)
            else:
                if flags&8:raise ValueError('stored descriptor stream unsupported')
                if csize>32*1024**2 or position+csize>len(raw):raise ValueError('stored member incomplete/large')
                value=raw[position:position+csize];position+=csize
            compressed=position-data_start
            if flags&8:
                if raw[position:position+4]==b'PK\x07\x08':position+=4
                if len(raw)-position<12:raise ValueError('incomplete descriptor')
                crc,csize,usize=struct.unpack_from('<3I',raw,position);position+=12
            if csize!=compressed or usize!=len(value) or crc!=zlib.crc32(value):raise ValueError('local size/CRC mismatch')
            total+=len(value)
            if total>512*1024**2:raise ValueError('512MiB decoded budget')
            retained=name.startswith('serving-wire-capacity9/output/control/') and not name.endswith('/')
            if retained:
                target=destination/'members'/path;target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as stream:stream.write(value)
            entries.append(dict(name=name,offset=start,next_offset=position,compressed_bytes=csize,bytes=usize,
                                crc32=f'{crc:08x}',sha256=hashlib.sha256(value).hexdigest(),retained=retained))
        except (ValueError,zlib.error) as error:
            stop=dict(offset=start,reason=str(error));break
    report=dict(scope='CRC_VERIFIED_LOCAL_MEMBERS_ONLY_NOT_COMPLETE_ARCHIVE',archive_sha256=EXPECTED,
                archive_bytes=len(raw),complete_archive=False,central_directory_verified=False,
                entries=entries,verified_members=len(entries),decoded_bytes=total,stopped=stop,
                elapsed_seconds=time.monotonic()-started,remote_calls=0)
    (destination/'RECOVERY.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='entries'},indent=2))

if __name__=='__main__':recover(HERE/'evidence/partial-members')
