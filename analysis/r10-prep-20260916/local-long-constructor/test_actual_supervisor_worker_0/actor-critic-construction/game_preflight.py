"""Stdlib-only exact public LS20 identity gate; never imports game source."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

DEFAULT_ENVIRONMENTS = Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files')
GAME_ID = 'ls20-9607627b'
FILES = {
    'ls20.py': (104880, '9607627b3d85bdb21ff28b40fefaa6a873691175981c6351b7e5a6637416f9d5'),
    'metadata.json': (298, '181e2f8a540962461a1c799160bdd11963d7276d6c42a368324dd84f3573803f'),
}
PUBLIC_MANIFEST_SHA256 = '842477b6336e4f556be999d6c9b76664238a5c7d991740263abf72fc1f864a35'


def validate(environments=DEFAULT_ENVIRONMENTS):
    root = Path(environments)
    if not root.is_absolute() or '..' in root.parts:
        raise ValueError('game root must be an absolute non-traversing path')
    selected = root / 'ls20' / '9607627b'
    for directory in [*reversed(selected.parents), selected]:
        if not stat.S_ISDIR(directory.lstat().st_mode):
            raise ValueError('game path is missing, symlinked or not a directory: ' + str(directory))
    if {p.name for p in (root / 'ls20').iterdir()} != {'9607627b'}:
        raise ValueError('ambiguous LS20 version directory')
    if {p.name for p in selected.iterdir()} != set(FILES):
        raise ValueError('unexpected or ambiguous selected game files')
    hashes = {}
    metadata = None
    for name, (size, digest) in FILES.items():
        path = selected / name
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, 'rb') as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size != size:
                raise ValueError('game source type/size mismatch: ' + name)
            raw = handle.read(size + 1)
        actual = hashlib.sha256(raw).hexdigest()
        if actual != digest:
            raise ValueError('game source digest mismatch: ' + name)
        hashes[name] = actual
        if name == 'metadata.json':
            metadata = json.loads(raw)
    if not isinstance(metadata, dict) or metadata.get('game_id') != GAME_ID or metadata.get('local_dir') != 'environment_files/ls20/9607627b':
        raise ValueError('game metadata identity mismatch')
    return {'status': 'passed', 'game_id': GAME_ID, 'environments': str(root),
            'selected_directory': str(selected), 'files': hashes,
            'public_manifest_sha256': PUBLIC_MANIFEST_SHA256,
            'scope': 'exact public bytes; no model request or game import'}


def qualify(receipt, environments=DEFAULT_ENVIRONMENTS):
    receipt = Path(receipt)
    try:
        result = validate(environments)
    except BaseException as error:
        receipt.write_text(json.dumps({'status': 'failed', 'game_id': GAME_ID,
                                      'error': f'{type(error).__name__}: {error}',
                                      'model_request_admitted': False}) + '\n')
        raise
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    return result
