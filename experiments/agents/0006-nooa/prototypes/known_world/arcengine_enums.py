"""PROTOTYPE: sandbox-compatible records for otherwise unchanged ARCEngine code."""

from __future__ import annotations

from enum import Enum, auto
from typing import Any

MAX_REASONING_BYTES = 16 * 1024


class BlockingMode(Enum):
    NOT_BLOCKED = auto()
    BOUNDING_BOX = auto()
    PIXEL_PERFECT = auto()


class InteractionMode(Enum):
    TANGIBLE = auto()
    INTANGIBLE = auto()
    INVISIBLE = auto()
    REMOVED = auto()


class GameState(str, Enum):
    NOT_PLAYED = "NOT_PLAYED"
    NOT_FINISHED = "NOT_FINISHED"
    WIN = "WIN"
    GAME_OVER = "GAME_OVER"


class SimpleAction:
    def __init__(self, game_id: str = "", **_: Any) -> None:
        self.game_id = game_id

    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> "SimpleAction":
        return cls(**data)


class ComplexAction(SimpleAction):
    def __init__(self, game_id: str = "", x: int = 0, y: int = 0, **_: Any) -> None:
        super().__init__(game_id)
        if not 0 <= x <= 63 or not 0 <= y <= 63:
            raise ValueError("x and y must be inside the 64x64 frame")
        self.x = x
        self.y = y


class GameAction(Enum):
    RESET = (0, SimpleAction)
    ACTION1 = (1, SimpleAction)
    ACTION2 = (2, SimpleAction)
    ACTION3 = (3, SimpleAction)
    ACTION4 = (4, SimpleAction)
    ACTION5 = (5, SimpleAction)
    ACTION6 = (6, ComplexAction)
    ACTION7 = (7, SimpleAction)

    def __init__(self, action_id: int, action_type: type[SimpleAction]) -> None:
        self._value_ = action_id
        self.action_type = action_type
        self.action_data = action_type()

    def is_simple(self) -> bool:
        return self.action_type is SimpleAction

    def is_complex(self) -> bool:
        return self.action_type is ComplexAction

    def validate_data(self, data: dict[str, Any]) -> bool:
        self.action_type.model_validate(data)
        return True

    def set_data(self, data: dict[str, Any]) -> SimpleAction:
        self.action_data = self.action_type(**data)
        return self.action_data

    @classmethod
    def from_id(cls, action_id: int) -> "GameAction":
        for action in cls:
            if action.value == action_id:
                return action
        raise ValueError(f"No GameAction with id {action_id}")

    @classmethod
    def from_name(cls, name: str) -> "GameAction":
        return cls[name.upper()]


class ActionInput:
    def __init__(
        self,
        id: GameAction | int = GameAction.RESET,
        data: dict[str, Any] | None = None,
        reasoning: Any = None,
    ) -> None:
        self.id = id if isinstance(id, GameAction) else GameAction.from_id(id)
        self.data = {} if data is None else data
        self.reasoning = reasoning


class FrameData:
    def __init__(self, **values: Any) -> None:
        self.game_id = values.get("game_id", "")
        self.frame = values.get("frame", [])
        self.state = values.get("state", GameState.NOT_PLAYED)
        self.levels_completed = values.get("levels_completed", 0)
        self.win_levels = values.get("win_levels", 0)
        self.action_input = values.get("action_input", ActionInput())
        self.guid = values.get("guid")
        self.full_reset = values.get("full_reset", False)
        self.available_actions = values.get("available_actions", [])

    def is_empty(self) -> bool:
        return not self.frame


class FrameDataRaw(FrameData):
    pass


class PlaceableArea:
    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        x_scale: int = 1,
        y_scale: int = 1,
    ) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.x_scale = x_scale
        self.y_scale = y_scale
