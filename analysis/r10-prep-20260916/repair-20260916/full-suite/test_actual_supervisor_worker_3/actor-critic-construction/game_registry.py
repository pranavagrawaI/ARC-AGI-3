"""Stdlib-only exact public game identity gate over a pinned table; never imports game source.

Same predicate as the R4 `game_preflight.py` LS20 gate (which stays byte-identical and still runs in
bootstrap), generalized over the games this package may play. Every entry is bound to the public
manifest 842477b6... by size and sha256; a game absent from this table cannot be selected.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

DEFAULT_ENVIRONMENTS = Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files')
PUBLIC_MANIFEST_SHA256 = '842477b6336e4f556be999d6c9b76664238a5c7d991740263abf72fc1f864a35'
REGISTRY = {
    'ls20': ('9607627b', {
        'ls20.py': (104880, '9607627b3d85bdb21ff28b40fefaa6a873691175981c6351b7e5a6637416f9d5'),
        'metadata.json': (298, '181e2f8a540962461a1c799160bdd11963d7276d6c42a368324dd84f3573803f'),
    }),
    's5i5': ('18d95033', {
        's5i5.py': (168955, '18d950333a006e984941ceaa16cdc4dc8dca8ca26cb82194198759089aac5965'),
        'metadata.json': (302, '1b4f3bfa7472ae0d9071f83183ecdcde04fa041e1530d868964829b5f7ed4a68'),
    }),
    'cd82': ('fb555c5d', {
        'cd82.py': (34273, 'fb555c5ddf0a3b742188646bb28d2f57cb98cf2d3263c594032b45ec8fce2a86'),
        'metadata.json': (292, 'ccfe95990d5a9c5df2cfe134452abd3397d63ede9c22afe02584f3b2a5184c9f'),
    }),
}


def game_id(game):
    if game not in REGISTRY:
        raise KeyError('game is not pinned in the registry: ' + repr(game))
    return f'{game}-{REGISTRY[game][0]}'


def validate(game, environments=DEFAULT_ENVIRONMENTS):
    version, files = REGISTRY[game]
    identity = game_id(game)
    root = Path(environments)
    if not root.is_absolute() or '..' in root.parts:
        raise ValueError('game root must be an absolute non-traversing path')
    selected = root / game / version
    for directory in [*reversed(selected.parents), selected]:
        if not stat.S_ISDIR(directory.lstat().st_mode):
            raise ValueError('game path is missing, symlinked or not a directory: ' + str(directory))
    if {p.name for p in (root / game).iterdir()} != {version}:
        raise ValueError(f'ambiguous {game.upper()} version directory')
    if {p.name for p in selected.iterdir()} != set(files):
        raise ValueError('unexpected or ambiguous selected game files')
    hashes = {}
    metadata = None
    for name, (size, digest) in files.items():
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
    if not isinstance(metadata, dict) or metadata.get('game_id') != identity or metadata.get('local_dir') != f'environment_files/{game}/{version}':
        raise ValueError('game metadata identity mismatch')
    return {'status': 'passed', 'game_id': identity, 'environments': str(root),
            'selected_directory': str(selected), 'files': hashes,
            'public_manifest_sha256': PUBLIC_MANIFEST_SHA256,
            'scope': 'exact public bytes; no model request or game import'}


def qualify(receipt, game, environments=DEFAULT_ENVIRONMENTS):
    receipt = Path(receipt)
    try:
        result = validate(game, environments)
    except BaseException as error:
        receipt.write_text(json.dumps({'status': 'failed', 'game_id': game_id(game) if game in REGISTRY else game,
                                      'error': f'{type(error).__name__}: {error}',
                                      'model_request_admitted': False}) + '\n')
        raise
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    return result
