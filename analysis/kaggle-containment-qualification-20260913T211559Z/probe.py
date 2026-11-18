"""Synthetic containment assessor; never imported into sterile runner."""
import hashlib
import accelerator

import json
import os
import platform
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import sysconfig
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = HERE/'runner.py'
RUNTIME_FILES = ('os.py', 'stat.py', 'types.py', 'enum.py', 'genericpath.py',
                 'posixpath.py', 'keyword.py', 'operator.py', 'reprlib.py',
                 'functools.py', 'warnings.py', 'copyreg.py')
RUNTIME_DIRS = ('json', 'encodings', 're', 'collections')
OUTPUT_CAP = 8192
IDENTITY = 'kaggle-containment-qualification-20260913T211559Z'
ASSET_NAMES = ('runner.py', 'probe.py', 'accelerator.py', 'audit_remote.py')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_relative(name):
    p = Path(name)
    if p.is_absolute() or not p.parts or '..' in p.parts or str(p) != name:
        raise ValueError('unsafe rootfs path')
    return p


def validate_tree(root, expected, *, require_root=True, test_link=None):
    if root.is_symlink() or root.stat().st_mode & 0o022:
        raise RuntimeError('UNSAFE_ROOT')
    observed = set()
    directories = {str(parent) for name in expected for parent in Path(name).parents if str(parent) != '.'}
    for p in [root, *root.rglob('*')]:
        s = p.lstat()
        if require_root and s.st_uid != 0:
            raise RuntimeError('NON_ROOT_OWNER')
        if p.is_symlink():
            if test_link is None or p != root/'work/escape-link' or os.readlink(p) != test_link:
                raise RuntimeError('ROOTFS_SYMLINK')
            observed.add(str(p.relative_to(root)))
            continue
        if s.st_mode & 0o022:
            raise RuntimeError('WRITABLE_ROOTFS')
        if stat.S_ISREG(s.st_mode):
            if s.st_nlink != 1:
                raise RuntimeError('ROOTFS_HARDLINK')
            observed.add(str(p.relative_to(root)))
        elif stat.S_ISDIR(s.st_mode):
            if p != root and str(p.relative_to(root)) not in directories:
                raise RuntimeError('ROOTFS_DIRECTORY_ALLOWLIST_MISMATCH')
        else:
            raise RuntimeError('ROOTFS_SPECIAL_FILE')
    if observed != set(expected):
        raise RuntimeError('ROOTFS_ALLOWLIST_MISMATCH')
    for name, expected_hash in expected.items():
        if name == 'work/escape-link' and test_link is not None:
            if expected_hash != 'synthetic-test-symlink:'+test_link:
                raise RuntimeError('ROOTFS_LINK_MANIFEST_MISMATCH')
        elif digest(root/name) != expected_hash:
            raise RuntimeError('ROOTFS_HASH_MISMATCH')


def rootfs(root, source, mode, args, *, require_root=True, test_link=None):
    if mode not in ('candidate', 'inspection'):
        raise ValueError('bad mode')
    if not isinstance(args, list) or len(json.dumps(args, allow_nan=False)) > 16384:
        raise ValueError('bad args')
    root.mkdir(mode=0o755)
    stdlib = Path(sysconfig.get_path('stdlib'))
    sources = [stdlib/name for name in RUNTIME_FILES]
    for name in RUNTIME_DIRS:
        sources += sorted((stdlib/name).rglob('*.py'))
    expected = {}
    for src in sources:
        if src.is_symlink() or not src.is_file() or 'site-packages' in src.parts:
            raise RuntimeError('UNSAFE_RUNTIME_SOURCE')
        name = 'runtime/'+str(src.relative_to(stdlib))
        validate_relative(name)
        dst = root/name
        dst.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        shutil.copyfile(src, dst)
        dst.chmod(0o444)
        expected[name] = digest(dst)
    # Interpreter/extension shared objects are already mapped by fresh exec;
    # no interpreter executable, host library tree or device is copied/mounted.
    files = {'work/candidate.py': source,
             'work/args.json': json.dumps({'mode': mode, 'args': args}, allow_nan=False)}
    if mode == 'inspection':
        files['work/prefix.json'] = json.dumps({'visible': ['SYNTHETIC_VISIBLE_PREFIX']})
    for name, content in files.items():
        validate_relative(name)
        dst = root/name
        dst.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        dst.write_text(content)
        dst.chmod(0o444)
        expected[name] = digest(dst)
    if test_link:
        (root/'work/escape-link').symlink_to(test_link)
        expected['work/escape-link'] = 'synthetic-test-symlink:'+test_link
    for p in [root, *[p for p in root.rglob('*') if p.is_dir()]]:
        p.chmod(0o555)
    validate_tree(root, expected, require_root=require_root, test_link=test_link)
    return expected


def bounded(command, *, timeout=4, output_cap=OUTPUT_CAP, extra_fds=(), env=None):
    if timeout <= 0 or type(output_cap) is not int or output_cap < 1:
        raise ValueError('INVALID_PROCESS_BOUNDS')
    start = time.monotonic()
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, close_fds=True,
        pass_fds=extra_fds, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
        env={'PATH': '/usr/bin:/bin', 'LANG': 'C'} if env is None else env)
    buffers = {process.stdout.fileno(): bytearray(), process.stderr.fileno(): bytearray()}
    status = 'completed'
    try:
        with selectors.DefaultSelector() as selector:
            for fd in buffers:
                selector.register(fd, selectors.EVENT_READ)
            total = 0
            while selector.get_map():
                if time.monotonic()-start >= timeout:
                    status = 'timeout'
                    break
                for key, _ in selector.select(.025):
                    chunk = os.read(key.fd, min(4096, output_cap-total+1))
                    if not chunk:
                        selector.unregister(key.fd)
                        continue
                    buffers[key.fd].extend(chunk)
                    total += len(chunk)
                    if total > output_cap:
                        status = 'output_limit'
                        break
                if status != 'completed':
                    break
    finally:
        if process.poll() is None:
            # EOF may precede exit; wait briefly only on the completed path.
            if status == 'completed':
                try:
                    process.wait(timeout=max(.001, timeout-(time.monotonic()-start)))
                except subprocess.TimeoutExpired:
                    status = 'timeout'
        # The session may still own children after its leader exits. Always
        # clean the invocation-owned group, including exceptional paths.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        stdout = buffers[process.stdout.fileno()].decode(errors='replace')
        stderr = buffers[process.stderr.fileno()].decode(errors='replace')
        process.stdout.close()
        process.stderr.close()
    return {'status': status, 'returncode': process.returncode, 'stdout': stdout,
            'stderr': stderr, 'elapsed_s': time.monotonic()-start,
            'reaped': process.poll() is not None}


def valid_seal(stderr):
    try:
        startup = json.loads(stderr)
        seal = startup['sealed']
        return (seal.get('uid') == [65534]*3
            and seal.get('gid') == [65534]*3 and seal.get('groups') == []
            and seal.get('capabilities_zero') is True and seal.get('bounding_dropped', 0) >= 38
            and seal.get('ambient_cleared') is True and seal.get('cwd') == '/'
            and seal.get('parent_death_signal') == signal.SIGKILL
            and seal.get('no_new_privs') is True and seal.get('seccomp') == 'native-allowlist')
    except (ValueError, TypeError, KeyError, AttributeError):
        return False


def assess(receipt, expected, status='completed'):
    # Exact output only: a forged PASS/JSON line cannot establish qualification.
    if receipt['status'] != status or not receipt['reaped']:
        return False
    if receipt['returncode'] != (0 if status == 'completed' else -signal.SIGKILL):
        return False
    try:
        value = json.loads(receipt['stdout']) if status == 'completed' else expected
    except (ValueError, TypeError):
        return False
    return value == expected and valid_seal(receipt['stderr'])


NVIDIA_SMI_PATHS = ('/opt/bin/nvidia-smi', '/usr/bin/nvidia-smi',
                    '/usr/local/bin/nvidia-smi', '/bin/nvidia-smi')


def discover_accelerator():
    return accelerator.discover(bounded, NVIDIA_SMI_PATHS)


def snapshot():
    def read(path):
        p = Path(path)
        return p.read_text()[:20000] if p.exists() else None
    result = {'platform': platform.platform(), 'uname': list(platform.uname()),
              'python': sys.version, 'executable': sys.executable,
              'uid': list(os.getresuid()), 'gid': list(os.getresgid()),
              'groups': os.getgroups(), 'os_release': read('/etc/os-release'),
              'proc_status': '\n'.join(l for l in (read('/proc/self/status') or '').splitlines()
                    if l.startswith(('Uid:', 'Gid:', 'Groups:', 'Cap', 'NoNewPrivs:', 'Seccomp:'))),
              'mountinfo': read('/proc/self/mountinfo'),
              'mapped_runtime_files': sorted({l.split()[-1] for l in (read('/proc/self/maps') or '').splitlines()
                    if len(l.split()) >= 6 and l.split()[-1].startswith('/')})}
    result['accelerator'] = discover_accelerator()
    result['sterile_runtime_imports'] = bounded([sys.executable, '-I', '-S', '-B', '-c',
        'import ctypes, json, os, platform, resource, signal, socket, subprocess, sys, time; from pathlib import Path; print("IMPORTS_OK")'],
        timeout=5, output_cap=4096)
    result['seccomp_dependency'] = 'native BPF via libc; libseccomp is not required'
    stdlib = Path(sysconfig.get_path('stdlib'))
    result['rootfs_runtime_sources'] = {'stdlib': str(stdlib), 'required_files': {name: (stdlib/name).is_file() for name in RUNTIME_FILES},
                                      'required_directories': {name: (stdlib/name).is_dir() for name in RUNTIME_DIRS}}
    # Only explicit nonsecret image identity facts; no environment dump.
    result['image_identity'] = {k: os.environ[k] for k in ('KAGGLE_DOCKER_IMAGE', 'KAGGLE_IMAGE_VERSION') if k in os.environ}
    result['requested_image'] = 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461'
    result['image_pin'] = 'Runtime cannot attest container digest; reconcile requested pin with provider readback metadata'
    return result


def cases(host_path, fd, mode, parent_pid, port):
    prefix = {'visible': ['SYNTHETIC_VISIBLE_PREFIX']} if mode == 'inspection' else None
    filesystem = f'''def run(*args):
 import os, json
 results = {{}}
 for name,path in {{'absolute':{host_path!r}, 'traversal':'/../../'+{host_path!r}.lstrip('/'), 'symlink':'/work/escape-link', 'proc':'/proc/{parent_pid}/root'+{host_path!r}, 'proc_environ':'/proc/{parent_pid}/environ', 'corpus':'/work/corpus.json', 'future':'/work/future.json', 'verifier':'/work/verifier.py', 'device':'/dev/nvidia0'}}.items():
  try:
   with open(path) as f: results[name] = f.read(256)
  except OSError: results[name] = 'denied'
 try:
  x = os.open('host-canary', os.O_RDONLY, dir_fd={fd}); results['openat_fd'] = os.read(x,256).decode(); os.close(x)
 except OSError: results['openat_fd'] = 'denied'
 try:
  os.fchdir({fd}); results['fchdir'] = 'escaped'
 except OSError: results['fchdir'] = 'denied'
 results['cwd'] = os.getcwd()
 try:
  results['prefix'] = json.load(open('/work/prefix.json'))
 except OSError: results['prefix'] = None
 return results
'''
    expected_fs = {k: 'denied' for k in ('absolute', 'traversal', 'symlink', 'proc', 'proc_environ', 'corpus', 'future', 'verifier', 'device', 'openat_fd', 'fchdir')}
    expected_fs.update(cwd='/', prefix=prefix)
    inherited = f'''def run(*args):
 import os
 result = {{'stdin': os.read(0,100).decode(), 'env': dict(os.environ), 'extra_fds': []}}
 for fd in range(3,128):
  try:
   os.fstat(fd); result['extra_fds'].append(fd)
  except OSError: pass
 return result
'''
    syscalls = '''def run(*args):
 import ctypes, os, socket, subprocess
 libc = ctypes.CDLL(None, use_errno=True)
 result = {}
 # Native numeric calls with invalid/null args. EPERM (not EFAULT/EINVAL)
 # proves filter refusal before any action. No child/memory bomb.
 for name,nr in {'socket':41,'connect':42,'ptrace':101,'process_vm_readv':310,'pidfd_getfd':438,'mount':165,'chroot':161,'setns':308,'unshare':272,'setuid':105,'setgid':106,'capset':126,'prctl':157,'clone':56,'fork':57,'vfork':58,'clone3':435,'execve':59,'execveat':322,'io_uring_setup':425,'open_by_handle_at':304,'prlimit64':302,'fcntl':72,'sched_getaffinity':204,'x32_getpid':0x40000027}.items():
  ctypes.set_errno(0)
  rc = libc.syscall(ctypes.c_long(nr),ctypes.c_long(-1),ctypes.c_long(0),ctypes.c_long(0),ctypes.c_long(0),ctypes.c_long(0),ctypes.c_long(0))
  result[name] = [rc,ctypes.get_errno()]
 try:
  subprocess.run(['/runtime/python','-c','pass'],timeout=1); result['subprocess']='allowed'
 except OSError: result['subprocess']='denied'
 try:
  sock=socket.socket(); sock.settimeout(.1); sock.connect(('127.0.0.1',PORT)); result['loopback']='allowed'; sock.close()
 except OSError: result['loopback']='denied'
 return result
'''.replace('PORT', str(port))
    names = ('socket connect ptrace process_vm_readv pidfd_getfd mount chroot setns unshare setuid setgid capset prctl clone fork vfork clone3 execve execveat io_uring_setup open_by_handle_at prlimit64 fcntl sched_getaffinity x32_getpid').split()
    return [
        ('benign', 'def run(x):\n import json\n return {"answer": x["a"] + x["b"], "roundtrip": json.loads(json.dumps(x))}\n', [{'a': 2, 'b': 3}], {'answer': 5, 'roundtrip': {'a':2,'b':3}}, 'completed'),
        ('filesystem', filesystem, [], expected_fs, 'completed'),
        ('inherited', inherited, [], {'stdin':'', 'env':{'LANG':'C','PYTHONDONTWRITEBYTECODE':'1'}, 'extra_fds':[]}, 'completed'),
        ('syscalls', syscalls, [], {**{n:[-1,1] for n in names}, 'subprocess':'denied', 'loopback':'denied'}, 'completed'),
        ('readonly', 'def run(*args):\n try:\n  open("/work/new-file","w").write("X")\n  return "writable"\n except OSError:\n  return "denied"\n', [], 'denied', 'completed'),
        ('wall', 'def run(*args):\n import time\n time.sleep(10)\n', [], None, 'timeout'),
        ('output', 'def run(*args):\n import os\n for _ in range(32): os.write(1,b"x"*4096)\n', [], None, 'output_limit'),
        ('forged', 'def run(*args):\n print("PASS")\n return {"qualified":True}\n', [], 'not-a-verdict', 'completed'),
    ]


def finalize(result, owned_base):
    result['required_cases'] = 16
    result['completed_cases'] = len(result['cases'])
    result['passed_cases'] = sum(c['check'] for c in result['cases'])
    result['failed_cases'] = result['completed_cases'] - result['passed_cases']
    result['unexercised_cases'] = result['required_cases'] - result['completed_cases']
    result['cleanup_complete'] = owned_base is None or not owned_base.exists()
    if not result['cleanup_complete']:
        result['classification'] = 'CLEANUP_FAILED'
    elif result['classification'] == 'QUALIFIED_SYNTHETIC_PROBE_ONLY' and (result['completed_cases'] != 16 or result['passed_cases'] != 16):
        result['classification'] = 'INCOMPLETE_QUALIFICATION'


def main(output, local=False):
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    owned_base = None
    result = {'classification':'STARTED', 'cases':[], 'model_loaded':False,
              'identity': IDENTITY, 'source_sha256': {name: digest(HERE/name) for name in ASSET_NAMES},
              'stage': 'platform_diagnostics',
              'requested_accelerator':'NvidiaRtxPro6000' if not local else None,
              'local_user_namespace': local}
    def save():
        temporary = output/'result.tmp'
        temporary.write_text(json.dumps(result, indent=2)+'\n')
        temporary.replace(output/'result.json')
    save()
    try:
        info = snapshot()
        (output/'platform.json').write_text(json.dumps(info,indent=2)+'\n')
        import runner
        runner.qualify()
        if not local:
            try:
                result['observed_accelerator'] = accelerator.parse_device(info['accelerator'])
            except (ValueError, KeyError, TypeError) as error:
                raise RuntimeError('TARGET_ACCELERATOR_NOT_OBSERVED:' + str(error)) from error
        # Separate exec runner; parent canaries are created only in this assessor.
        import socket
        result['stage'] = 'parent_canary_setup'
        with tempfile.TemporaryDirectory(prefix='containment-q-') as temporary:
            base = Path(temporary)
            owned_base = base
            base.chmod(0o755)
            hidden = base/'host-canary'
            canary = 'SYNTHETIC-'+os.urandom(24).hex()
            hidden.write_text(canary)
            hidden.chmod(0o644)  # readable by dropped UID IF confinement escapes
            fd = None
            listener = None
            try:
                fd = os.open(base, os.O_RDONLY | os.O_DIRECTORY)
                listener = socket.socket()
                listener.bind(('127.0.0.1', 0)); listener.listen(1); listener.settimeout(.02)
                for mode in ('candidate', 'inspection'):
                    for name, source, args, expected, status in cases(str(hidden), fd, mode, os.getpid(), listener.getsockname()[1]):
                        result['stage'] = mode + ':' + name + ':rootfs'
                        save()
                        if time.monotonic()-start > 120:
                            raise RuntimeError('GLOBAL_ADMISSION_DEADLINE')
                        root = base/(mode+'-'+name)
                        manifest = rootfs(root, source, mode, args, test_link=str(hidden) if name=='filesystem' else None)
                        (output/(mode+'-'+name+'-rootfs.json')).write_text(json.dumps(manifest,indent=2)+'\n')
                        # Reconcile at admission, not only when constructing or
                        # after execution. Manifest stays in the trusted host.
                        validate_tree(root, manifest, test_link=str(hidden) if name=='filesystem' else None)
                        receipt = bounded([sys.executable, '-I', '-S', '-B', str(RUNNER), str(root), str(os.getpid())], timeout=.5 if name=='wall' else 4,
                                          extra_fds=(fd,))
                        check = assess(receipt, expected, status)
                        if name == 'forged':
                            check = not check and valid_seal(receipt['stderr']) and receipt['status']=='completed' and receipt['returncode']==0 and receipt['stdout']=='PASS\n{"qualified": true}\n'
                        no_connection = True
                        try:
                            connection, _ = listener.accept(); connection.close(); no_connection=False
                        except TimeoutError:
                            pass
                        untouched = hidden.read_text() == canary
                        try:
                            validate_tree(root, manifest, test_link=str(hidden) if name=='filesystem' else None)
                            rootfs_unchanged = True
                        except (OSError, RuntimeError) as error:
                            rootfs_unchanged = False
                            receipt['rootfs_validation_error'] = type(error).__name__ + ':' + str(error)
                        receipt.update(mode=mode, name=name, expected=expected, expected_status=status,
                                       check=check and no_connection and untouched and rootfs_unchanged and canary not in receipt['stdout']+receipt['stderr'],
                                       rootfs_unchanged=rootfs_unchanged,
                                       parent_canary_unchanged=untouched, parent_no_loopback_connection=no_connection)
                        result['cases'].append(receipt)
                        save()
                        if not receipt['check']:
                            raise RuntimeError('CASE_FAILED:'+mode+':'+name)
            finally:
                try:
                    if fd is not None:
                        os.close(fd)
                finally:
                    if listener is not None:
                        listener.close()
        result['classification'] = 'QUALIFIED_SYNTHETIC_PROBE_ONLY'
        result['stage'] = 'complete'
    except Exception as error:
        result['classification'] = 'UNSUPPORTED_OR_QUALIFICATION_FAILED'
        result['error'] = type(error).__name__+':'+str(error)
    finally:
        result['elapsed_s'] = time.monotonic()-start
        finalize(result, owned_base)
        save()
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
    return 0 if result['classification']=='QUALIFIED_SYNTHETIC_PROBE_ONLY' else 2

if __name__ == '__main__':
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('GLOBAL_150S_LIMIT')))
    signal.alarm(150)
    raise SystemExit(main(Path(sys.argv[1]), '--local' in sys.argv[2:]))
