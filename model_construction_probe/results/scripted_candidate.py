
def initialize(o):
    return [{"candidate": "boundary", "state": {"x": o["x"], "limit": o["limit"]}}]

def predict_one(candidate, s, action):
    x = min(s["x"] + 1, s["limit"])
    return [{"state": {"x": x, "limit": s["limit"]}, "claims": {"x": [x]}}]

def observe_one(candidate, s, o):
    return s
