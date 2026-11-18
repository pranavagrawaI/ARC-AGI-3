
"""World model for the grid game."""
import numpy as np

# Color constants
C_BG = 4       # blue background
C_GREEN = 3    # green region
C_BORDER = 5   # orange/green border
C_12 = 12      # gray (top of object)
C_9 = 9        # white (bottom of object)
C_0 = 0        # target part 1
C_1 = 1        # target part 2
C_BAR = 11     # status bar
C_MARK = 8     # status marks

def encode(grid):
    """Extract the latent state z from the 64x64 grid."""
    z = {}
    # Find the 12-block (top of object) - should be exactly 10 cells (2x5)
    rs, cs = np.where(grid == C_12)
    if len(rs) == 10:
        z['obj_r'] = int(rs.min())
        z['obj_c'] = int(cs.min())
    else:
        z['obj_r'] = None
        z['obj_c'] = None
    # Find the moving 9-block: the 3x5 block adjacent to (below) the 12-block
    if z['obj_r'] is not None:
        rs9, cs9 = np.where(grid == C_9)
        # The moving 9-block is a 3x5 block, with top row at obj_r+2 (or obj_r if overlapping)
        # Filter: within 10 rows AND within 10 cols of the 12-block
        mask = (np.abs(rs9 - z['obj_r']) <= 10) & (np.abs(cs9 - z['obj_c']) <= 10)
        if mask.any():
            rs9f, cs9f = rs9[mask], cs9[mask]
            # Should be exactly 15 cells (3x5)
            if len(rs9f) == 15:
                z['obj9_r'] = int(rs9f.min())
                z['obj9_c'] = int(cs9f.min())
            else:
                z['obj9_r'] = int(rs9f.min())
                z['obj9_c'] = int(cs9f.min())
        else:
            z['obj9_r'] = None
            z['obj9_c'] = None
    else:
        z['obj9_r'] = None
        z['obj9_c'] = None
    # Target position (0/1 cluster)
    rs0, cs0 = np.where(grid == C_0)
    if len(rs0) > 0:
        z['target_r'] = int(rs0.min())
        z['target_c'] = int(cs0.min())
    else:
        z['target_r'] = None
        z['target_c'] = None
    return z

def predict(z, action):
    """Predict next latent state after action. Object moves 5 cells in commanded direction."""
    z2 = dict(z)
    dr, dc = 0, 0
    if action == 'UP': dr = -5
    elif action == 'DOWN': dr = 5
    elif action == 'LEFT': dc = -5
    elif action == 'RIGHT': dc = 5
    if z.get('obj_r') is not None:
        z2['obj_r'] = z['obj_r'] + dr
        z2['obj_c'] = z['obj_c'] + dc
    if z.get('obj9_r') is not None:
        z2['obj9_r'] = z['obj9_r'] + dr
        z2['obj9_c'] = z['obj9_c'] + dc
    return z2

Z_SCHEMA = {
    'obj_r': 'int, top row of 12-block (object top)',
    'obj_c': 'int, left col of 12-block (object left)',
    'obj9_r': 'int, top row of 9-block (object bottom)',
    'obj9_c': 'int, left col of 9-block',
    'target_r': 'int, top row of 0/1 target',
    'target_c': 'int, left col of 0/1 target',
}
