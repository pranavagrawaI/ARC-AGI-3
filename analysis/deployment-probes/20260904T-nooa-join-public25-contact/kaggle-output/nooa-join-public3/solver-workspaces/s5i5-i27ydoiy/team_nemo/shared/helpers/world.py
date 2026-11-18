
"""World model for the hex-grid game. z = dict of entity latents."""
import numpy as np

Z_SCHEMA = {
    "tile": "dict or None: center of the 6x6 clickable tile pattern (rows 35..40, cols 35..40)",
    "tiles3": "list of (r,c) of color-3 cells outside the status rows",
    "player": "(r,c) of color-d diagonal cluster, or None",
    "p2": "second player marker (d cluster near 51..53) or None",
    "p3": "third marker cluster (d near 9..11) or None",
    "row63": "bool: status row 63 is color 3",
    "grid": "the full 64x64 array itself (lossy fallback)",
}

def encode(g):
    """g: 64x64 int array -> z dict."""
    z = {"grid": np.asarray(g)}
    nz = np.argwhere(g != 5)
    z["non5"] = [tuple(map(int, t)) for t in nz]
    # player = d-cluster cells
    dcells = list(map(tuple, np.argwhere(g == 13).astype(int)))
    z["d_cells"] = dcells
    ecells = list(map(tuple, np.argwhere(g == 14).astype(int)))
    z["e_cells"] = ecells
    return z

def z_equals(a, b, ignore=()):
    """compare z dicts on keys present in both, ignoring status band rows (0-8, 63)."""
    for k in a:
        if k in ignore or k not in b:
            continue
        if k == "grid":
            diff = np.argwhere(np.asarray(a[k]) != np.asarray(b[k]))
            diff = [tuple(t) for t in diff if 9 <= t[0] <= 62]
            return len(diff) == 0
        if a[k] != b[k]:
            return False
    return True
