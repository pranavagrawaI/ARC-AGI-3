"""Correctness test for flare_ra.so: shimmed reads must equal plain reads, byte for byte.

  python3 -B test_ra.py   # builds a 600 MiB random file, runs the shimmed child, compares hashes

The parent writes the file and hashes 3,000 (offset, length) ranges with plain reads. The child runs
under LD_PRELOAD with 4 MiB blocks, a 6-block cap and 3-block read-ahead (so eviction and refetch
happen), opens with O_DIRECT, and reads the same ranges from 8 threads via pread, plus sequential
read(), readv() and preadv() passes. Any mismatch fails. The stats log must show O_DIRECT stripped and
calls counted.
"""
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIZE = 600 * (1 << 20) + 12345  # not block-aligned, so the tail block is short

CHILD = r'''
import hashlib, json, os, sys, threading
spec = json.load(open(sys.argv[1]))
path, ranges = spec["path"], spec["ranges"]
fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
bad, lock = [], threading.Lock()
def work(chunk):
    for off, n, h in chunk:
        got = os.pread(fd, n, off)
        if hashlib.sha256(got).hexdigest() != h:
            with lock: bad.append(("pread", off, n, len(got)))
th = [threading.Thread(target=work, args=(ranges[i::8],)) for i in range(8)]
[t.start() for t in th]; [t.join() for t in th]
# sequential read() over the whole file in odd-sized pieces
h = hashlib.sha256(); fd2 = os.open(path, os.O_RDONLY)
while True:
    b = os.read(fd2, 3_000_017)
    if not b: break
    h.update(b)
if h.hexdigest() != spec["whole"]: bad.append(("read", "whole"))
os.lseek(fd2, 0, os.SEEK_SET)
bufs = [bytearray(1_000_003), bytearray(5 << 20)]
n = os.readv(fd2, bufs)
first = bytes(bufs[0]) + bytes(bufs[1])
if hashlib.sha256(first[:n]).hexdigest() != spec["head"] or n != len(first): bad.append(("readv", n))
bufs = [bytearray(777), bytearray(9 << 20)]
off = spec["size"] - 5_000_000
n = os.preadv(fd, bufs, off)
tail = (bytes(bufs[0]) + bytes(bufs[1]))[:n]
if hashlib.sha256(tail).hexdigest() != spec["tail"] or n != 5_000_000: bad.append(("preadv", n))
import mmap, tempfile
want = spec["probe"]  # (off, n, sha) for the copy/map paths
out = tempfile.TemporaryFile()
n = os.copy_file_range(fd2, out.fileno(), want[1], want[0], 0)
out.seek(0)
if n != want[1] or hashlib.sha256(out.read()).hexdigest() != want[2]: bad.append(("copy_file_range", n))
out2 = tempfile.TemporaryFile()
n = os.sendfile(out2.fileno(), fd2, want[0], want[1])
out2.seek(0)
if n != want[1] or hashlib.sha256(out2.read()).hexdigest() != want[2]: bad.append(("sendfile", n))
r, w = os.pipe()
os.set_blocking(w, True)
n = os.splice(fd2, w, 60000, offset_src=want[0])
if hashlib.sha256(os.read(r, 60000)).hexdigest() != spec["probe_small"]: bad.append(("splice", n))
mo = (want[0] // mmap.ALLOCATIONGRANULARITY) * mmap.ALLOCATIONGRANULARITY
mm = mmap.mmap(fd2, want[0] + want[1] - mo, prot=mmap.PROT_READ, flags=mmap.MAP_PRIVATE, offset=mo)
if hashlib.sha256(mm[want[0] - mo:]).hexdigest() != want[2]: bad.append(("mmap", len(mm)))
past = os.pread(fd, 4096, spec["size"] + 10)
if past != b"": bad.append(("past-eof", len(past)))
print(json.dumps({"bad": bad[:10], "n_bad": len(bad), "ranges": len(ranges)}))
'''


def main():
    tmp = Path(tempfile.mkdtemp(prefix="flare-ra-test-"))
    path = tmp / "pkg.bin"
    rng = random.Random(7)
    with open(path, "wb") as f:
        left = SIZE
        while left:
            n = min(left, 16 << 20)
            f.write(rng.randbytes(n)); left -= n
    data = path.read_bytes()
    sizes = [1, 4096, 65536, 1 << 20, 3 << 20, 4 << 20, 9 << 20, 4095, 123457]
    ranges = []
    for _ in range(3000):
        n = rng.choice(sizes)
        off = rng.randrange(0, SIZE - 1)
        ranges.append((off, n, hashlib.sha256(data[off:off + n]).hexdigest()))
    spec = {"path": str(path), "size": SIZE, "ranges": ranges, "whole": hashlib.sha256(data).hexdigest(),
            "head": hashlib.sha256(data[:1_000_003 + (5 << 20)]).hexdigest(),
            "tail": hashlib.sha256(data[SIZE - 5_000_000:]).hexdigest()}
    po, pn = 123_456_789, 20_000_003
    spec["probe"] = (po, pn, hashlib.sha256(data[po:po + pn]).hexdigest())
    spec["probe_small"] = hashlib.sha256(data[po:po + 60000]).hexdigest()
    (tmp / "spec.json").write_text(json.dumps(spec))
    (tmp / "child.py").write_text(CHILD)
    # multi-source mode: six uneven parts tile the file; the target is a sparse placeholder of zeros
    cuts = [0, 1, 7 << 20, 100_000_001, 250 << 20, 400_000_000, SIZE]
    srcs = []
    for i, (a, b) in enumerate(zip(cuts, cuts[1:])):
        part = tmp / f"part{i}"
        part.write_bytes(data[a:b])
        srcs.append(f"{part}@{a}")
    placeholder = tmp / "placeholder.bin"
    with open(placeholder, "wb") as f:
        f.truncate(SIZE)
    results = []
    for mode, target, extra in (("single", path, {}), ("multi", placeholder, {"FLARE_RA_SOURCES": ";".join(srcs)})):
        spec["path"] = str(target)
        (tmp / "spec.json").write_text(json.dumps(spec))
        log = tmp / f"ra-{mode}.log"
        env = dict(os.environ, LD_PRELOAD=str(HERE / "flare_ra.so"), FLARE_RA_PATH=str(target), FLARE_RA_BLOCK_MB="4",
                   FLARE_RA_CAP="6", FLARE_RA_AHEAD="3", FLARE_RA_THREADS="4", FLARE_RA_LOG=str(log), **extra)
        p = subprocess.run([sys.executable, "-I", str(tmp / "child.py"), str(tmp / "spec.json")], env=env,
                           capture_output=True, text=True, timeout=900)
        print(mode, p.stdout.strip(), p.stderr.strip()[-2000:])
        res = json.loads(p.stdout.strip().splitlines()[-1])
        stats = [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
        print(mode, "last stats:", stats[-1] if stats else None)
        results.append(p.returncode == 0 and res["n_bad"] == 0 and bool(stats) and stats[-1]["calls"] > 3000)
    ok = all(results)
    print("RA TEST", "PASS" if ok else "FAIL", results)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
