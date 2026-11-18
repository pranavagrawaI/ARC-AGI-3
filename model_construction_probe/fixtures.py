"""Invented diagnostic worlds. None are LS20, S5I5, or CD82 trace reproductions."""

ALTERNATING = '''
def initialize(o):
    return [{"candidate": "alternating", "state": {"length": o["length"], "phase": p}}
            for p in (0, 1)]

def predict_one(candidate, s, action):
    n = s["length"] + 1 + s["phase"]
    return [{"state": {"length": n, "phase": 1 - s["phase"]},
             "claims": {"length": [n]}}]

def observe_one(candidate, s, o):
    return s
'''

INDEPENDENT = '''
def initialize(o):
    return [{"candidate": "independent", "state": {"length": o["length"]}}]

def predict_one(candidate, s, action):
    return [{"state": {"length": s["length"] + d},
             "claims": {"length": [s["length"] + d]}} for d in (1, 2)]

def observe_one(candidate, s, o):
    return s
'''

WRONG_BUT_ASSIMILATES = '''
def initialize(o):
    return [{"candidate": "always_move", "state": {"x": o["x"]}}]

def predict_one(candidate, s, action):
    return [{"state": {"x": s["x"] + 1}, "claims": {"x": [s["x"] + 1]}}]

def observe_one(candidate, s, o):
    return {"x": o["x"]}
'''

BOUNDARY = '''
def initialize(o):
    return [{"candidate": "boundary", "state": {"x": o["x"], "limit": o["limit"]}}]

def predict_one(candidate, s, action):
    x = min(s["x"] + 1, s["limit"])
    return [{"state": {"x": x, "limit": s["limit"]}, "claims": {"x": [x]}}]

def observe_one(candidate, s, o):
    return s
'''

UNKNOWN = '''
def initialize(o):
    return [{"candidate": "unknown", "state": {}}]

def predict_one(candidate, s, action):
    return [{"state": {}, "claims": {}, "unresolved": ["transition mechanism"]}]

def observe_one(candidate, s, o):
    return s
'''

CORRELATED_FIELDS = '''
def initialize(o):
    return [{"candidate": "paired_effects", "state": {}}]

def predict_one(candidate, s, action):
    return [{"state": {}, "claims": {"x": [1], "meter": [9]}},
            {"state": {}, "claims": {"x": [0], "meter": [8]}}]

def observe_one(candidate, s, o):
    return s
'''

OCCLUSION = '''
def initialize(o):
    return [{"candidate": "remember_visible_terrain", "state": {"terrain": o.get("terrain")}}]

def predict_one(candidate, s, action):
    # The model claims no terrain observation while the cell is covered.
    if action == "cover":
        return [{"state": s, "claims": {}, "unresolved": ["terrain visibility"]}]
    if s["terrain"] is None:
        return [{"state": s, "claims": {}, "unresolved": ["terrain value"]}]
    return [{"state": s, "claims": {"terrain": [s["terrain"]]}}]

def observe_one(candidate, s, o):
    if "terrain" in o:
        s["terrain"] = o["terrain"]
    return s
'''

MUTATING_GLOBAL = '''
count = 0

def initialize(o):
    return [{"candidate": "test", "state": {"x": 0}}]

def predict_one(candidate, s, action):
    global count
    count += 1
    return [{"state": {"x": count}, "claims": {"x": [count]}}]

def observe_one(candidate, s, o):
    return s
'''
