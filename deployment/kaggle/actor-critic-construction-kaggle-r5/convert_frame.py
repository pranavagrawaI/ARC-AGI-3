from __future__ import annotations
from arcengine import FrameData, FrameDataRaw
def _convert_raw_frame_data(self, raw: FrameDataRaw | None) -> FrameData:
    if raw is None:
        raise ValueError('Received None frame data from environment')
    out = FrameData(game_id=raw.game_id, frame=[arr.tolist() for arr in raw.frame], state=raw.state, levels_completed=raw.levels_completed, win_levels=raw.win_levels, guid=raw.guid, full_reset=raw.full_reset, available_actions=raw.available_actions)
    return out
def convert(raw): return _convert_raw_frame_data(None, raw)
