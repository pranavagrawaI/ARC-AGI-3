
import numpy as np

def encode(grid):
    """Encode the 64x64 grid into a latent state dict."""
    # Track the 2-bar (row 53): which columns are 3
    bar = []
    for c in range(64):
        bar.append(int(grid[53, c]))
    # Track selection: check which mini-box has its border removed
    # Mini-boxes at rows 57-60, cols: e=18-21, f=26-29, 9=34-37, b=42-45
    # Selected = border cells are 4 (background) instead of the box color
    sel = None
    for name, r1, r2, c1, c2, color in [
        ('e', 57, 60, 18, 21, 14),
        ('f', 57, 60, 26, 29, 15),
        ('9', 57, 60, 34, 37, 9),
        ('b', 57, 60, 42, 45, 11),
    ]:
        # Check if border cells are 4 (removed) vs color (present)
        border_removed = (grid[r1, c1] == 4 and grid[r2, c2] == 4)
        if border_removed:
            sel = name
            break
    # 8x8 grid: track if any interior cell changed from 5
    grid_changed = 0
    for r in range(2, 6):
        for c1, c2 in [(18, 24), (26, 32), (34, 40), (42, 48)]:
            grid_changed += int(np.sum(grid[r, c1:c2] != 5))
    return {
        "bar": bar,
        "sel": sel if sel else "",
        "grid_non5": grid_changed,
    }


def predict(z, action):
    """Predict the next latent state given action."""
    z = dict(z)  # copy
    bar = list(z["bar"])
    sel = z["sel"]
    action = action.strip().upper()

    if action == "SPACEBAR":
        # Find rightmost 2 and change to 3
        changed = False
        for c in range(63, -1, -1):
            if bar[c] == 2:
                bar[c] = 3
                changed = True
                break
        # Clear selection
        sel = ""
        if not changed:
            # Bar is full — what happens? Unknown. No change for now.
            pass
    elif action.startswith("CLICK"):
        parts = action.split()
        if len(parts) == 3:
            x, y = int(parts[1]), int(parts[2])  # col, row
            # Check if click is on a mini-box
            for name, r1, r2, c1, c2, color in [
                ('e', 57, 60, 18, 21, 14),
                ('f', 57, 60, 26, 29, 15),
                ('9', 57, 60, 34, 37, 9),
                ('b', 57, 60, 42, 45, 11),
            ]:
                if r1 <= y <= r2 and c1 <= x <= c2:
                    # Toggle selection: if this box is selected, deselect; else select
                    if sel == name:
                        sel = ""
                    else:
                        sel = name
                    break
            # Click elsewhere: no effect (already established)
    elif action == "UNDO":
        pass  # no_impact (established)
    elif action == "RESET":
        # Unknown — would reset everything
        bar = [2] * 64
        sel = ""

    z["bar"] = bar
    z["sel"] = sel
    return z
