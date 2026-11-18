"""Queue notifications wake the actor; the host file supplies current state."""

import json


class StateDelivery:
    def __init__(self):
        self.delivered = None

    def prepare(self, notification, latest):
        if not latest or "turn" not in latest:
            return notification
        key = (latest["turn"], latest["step"])
        other = {k: v for k, v in notification.items() if k != "game_states"}
        if key == self.delivered and not any(other.values()):
            return None
        self.delivered = key
        return {**other, "game_states": [json.dumps(latest)]}
