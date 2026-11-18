import numpy as np

def parse(grid_rows):
    g = np.array([int(ch, 16) for row in grid_rows for ch in row]).reshape(64, 64)
    return g

def find_shapes(g):
    results = {}
    for c in [4, 5, 11]:
        m = (g == c)
        ys, xs = np.where(m)
        if len(ys) > 0:
            results[c] = {
                'count': int(len(ys)),
                'row_min': int(ys.min()), 'row_max': int(ys.max()),
                'col_min': int(xs.min()), 'col_max': int(xs.max()),
            }
    return results

def encode(grid_rows):
    g = parse(grid_rows)
    shapes = find_shapes(g)
    return {
        'shapes': shapes,
    }
