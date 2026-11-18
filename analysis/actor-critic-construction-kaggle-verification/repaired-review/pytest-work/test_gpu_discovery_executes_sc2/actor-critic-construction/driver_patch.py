"""Small restoration scope for the inherited qualification-driver module injection."""
class DriverPatch:
    def __init__(self):
        self.records = []

    def setitem(self, mapping, key, value):
        self.records.append((mapping, key, key in mapping, mapping.get(key)))
        mapping[key] = value

    def undo(self):
        while self.records:
            mapping, key, existed, value = self.records.pop()
            if existed:
                mapping[key] = value
            else:
                mapping.pop(key, None)
