from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from geometry import Direction, TilePos
from models import CorridorGeometry


@dataclass(frozen=True)
class PortRequirement:
    """Specifies the desired doorway placement for a special junction room."""

    center: Tuple[float, float]
    direction: Direction
    width: int
    inside_tiles: Tuple[TilePos, ...]
    outside_tiles: Tuple[TilePos, ...]
    source: str
    geometry: Optional[CorridorGeometry] = None
    room_index: Optional[int] = None
    port_index: Optional[int] = None
    corridor_idx: Optional[int] = None
    corridor_end: Optional[str] = None
