
import numpy as np
import collections

# Layout: 64x64. Col 0 is an "e"(14) overlay border.
#   rows 0-12: bg 3; rows 13-50: bg 4; rows 51-63: bg 3.
# Objects are 4x4 blocks of solid color on a 4-aligned grid:
#   block (br, bc) covers rows 4br..4br+3, cols 4bc..4bc+3.

def band_bg(row):
    if row < 13: return 3
    if row < 51: return 4
    return 3

def encode(g, level=0):
    blocks = {}
    for br in range(16):
        r0 = 4*br
        for bc in range(16):
            sub = g[r0:r0+4, 4*bc:4*bc+4]
            vals, cnts = np.unique(sub, return_counts=True)
            color = int(vals[np.argmax(cnts)])
            if color != band_bg(r0):
                blocks[f"{br},{bc}"] = color
    pal = collections.Counter(blocks.values())
    return {"blocks": dict(sorted(blocks.items())), "palette": dict(sorted(pal.items())),
            "level": level}

def blocks_of(g):
    """Only the block map (cheaper compare)."""
    z = encode(g)
    return z["blocks"]

def predict(z, action):
    # Unknown dynamics; stub returns z unchanged.
    return dict(z)
