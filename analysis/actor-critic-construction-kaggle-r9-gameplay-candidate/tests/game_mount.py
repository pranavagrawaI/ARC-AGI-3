"""Disposable mount namespace: real default Kaggle path, never checkout edits."""
from __future__ import annotations
import ctypes
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
PUBLIC = ROOT / 'analysis/actor-critic-construction-kaggle-verification-r4/public-assets/files'
PUBLIC_R9 = ROOT / 'analysis/actor-critic-construction-kaggle-verification-r9/public-assets/files'
# Exact public bytes staged from the competition (kaggle competitions download -f ...), verified
# against manifest 842477b6... in verification-r9/public-assets/VERIFICATION.json.
GAMES = {'ls20': ('9607627b', PUBLIC, ('ls20.py', 'metadata.json')),
         's5i5': ('18d95033', PUBLIC_R9 / 's5i5', ('s5i5.py', 'metadata.json')),
         'cd82': ('fb555c5d', PUBLIC_R9 / 'cd82', ('cd82.py', 'metadata.json'))}
RELATIVE = Path('kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files')
# Pinned official ARC-AGI-3-Agents checkout retained from a Kaggle export; the driver
# re-verifies every OFFICIAL_HASHES entry and the revision after copying it.
FRAMEWORK = ROOT / 'analysis/deployment-probes/20260904T-nooa-join-public3-depth/kaggle-output/ARC-AGI-3-Agents'


def copy_game(root):
    first = None
    for game, (version, source, names) in GAMES.items():
        target = Path(root) / RELATIVE / game / version
        target.mkdir(parents=True)
        for name in names:
            shutil.copyfile(source / name, target / name)
        first = first or target
    return first  # the ls20 directory, which the inherited mutate() fixtures target


def copy_framework(root):
    target = Path(root) / RELATIVE.parent / 'ARC-AGI-3-Agents'
    shutil.copytree(FRAMEWORK, target, symlinks=True, ignore=shutil.ignore_patterns('__pycache__'))
    return target


def enter(root, mutate=None, game='ls20'):
    """Only for a fresh test child already inside mapped-root user namespace.

    `game` selects which staged game directory `mutate` receives: the bootstrap gate
    checks ls20; the application gate checks the package's PLAY_GAME."""
    root = Path(root)
    root.mkdir()
    copy_game(root)
    copy_framework(root)
    version = GAMES[game][0]
    if mutate is not None:
        mutate(Path(root) / RELATIVE / game / version)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.unshare(0x00020000) != 0:  # CLONE_NEWNS
        raise OSError(ctypes.get_errno(), 'private mount namespace')
    subprocess.run(['mount', '--make-rprivate', '/'], check=True)
    # Keep the test interpreter/dependencies at unchanged absolute paths. These
    # are trusted-host fixture mounts, NOT generated-code containment mounts.
    for source in sorted(Path('/').iterdir(), key=lambda p: p.name != 'tmp'):
        if source.name not in {'home', 'tmp', 'usr', 'etc', 'dev', 'proc', 'opt', 'lib', 'lib64', 'bin', 'sbin'}:
            continue
        target = root / source.name
        if source.is_symlink():
            target.symlink_to(os.readlink(source))
        elif source.is_dir():
            target.mkdir()
            subprocess.run(['mount', '--rbind', str(source), str(target)], check=True)
    selected = root / RELATIVE
    subprocess.run(['mount', '--bind', str(selected), str(selected)], check=True)
    subprocess.run(['mount', '-o', 'remount,bind,ro', str(selected)], check=True)
    os.chroot(root)
    os.chdir('/')
