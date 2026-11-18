"""Unintegrated source transformation only; no new notebook/assignment/launch."""

GUARD = '''# Bind package imports before the notebook runpy entrypoint.
import importlib
def _verify_loaded_payload_modules():
    for _filename, _record in PAYLOAD.items():
        if not _filename.endswith('.py'):
            continue
        _module = sys.modules.get(_filename[:-3])
        if _module is None:
            continue
        _origin = getattr(_module, '__file__', None)
        _expected = (root / _filename).resolve()
        if _origin is None or pathlib.Path(_origin).resolve() != _expected:
            raise RuntimeError('package module origin mismatch: ' + _filename)
        if hashlib.sha256(_expected.read_bytes()).hexdigest() != _record['sha256']:
            raise RuntimeError('package module bytes mismatch: ' + _filename)
sys.path.insert(0, str(root))
_verify_loaded_payload_modules()
importlib.import_module('release_runner')
_verify_loaded_payload_modules()
'''


def apply(source):
    needle = "sys.argv = [str(root / 'supervisor.py'), str(root / 'config.json'), str(root / 'output'), '--assignment', str(root / 'assignment.json')]\nrunpy.run_path(str(root / 'supervisor.py'), run_name='__main__')\n"
    if source.count(needle) != 1:
        raise ValueError('exact reviewed bootstrap tail required')
    return source.replace(needle, GUARD + needle)
