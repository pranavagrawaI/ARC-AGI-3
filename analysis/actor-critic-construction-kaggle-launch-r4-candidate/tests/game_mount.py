"""Disposable mount namespace: real default Kaggle path, never checkout edits."""
from __future__ import annotations
import ctypes
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
PUBLIC = ROOT / 'analysis/actor-critic-construction-kaggle-verification-r4/public-assets/files'
RELATIVE = Path('kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files')


def copy_game(root):
    target = Path(root) / RELATIVE / 'ls20/9607627b'
    target.mkdir(parents=True)
    for name in ('ls20.py', 'metadata.json'):
        shutil.copyfile(PUBLIC / name, target / name)
    return target


def enter(root, mutate=None):
    """Only for a fresh test child already inside mapped-root user namespace."""
    root = Path(root)
    root.mkdir()
    game = copy_game(root)
    if mutate is not None:
        mutate(game)
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
