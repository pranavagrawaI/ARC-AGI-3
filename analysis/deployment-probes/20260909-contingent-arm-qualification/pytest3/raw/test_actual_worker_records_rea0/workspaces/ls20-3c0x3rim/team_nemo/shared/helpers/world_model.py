def encode(g): return {"cells": [(0,0)]}
def predict(z,a): return {"n":len(set(map(tuple,z["cells"])))}
