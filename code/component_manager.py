"""Component management utilities backed by a disjoint-set union structure."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Set


class DisjointSetUnion:
    """Disjoint set union with path compression and canonical minimum roots."""

    def __init__(self) -> None:
        self._parent: Dict[int, int] = {}

    def _ensure(self, item: int) -> None:
        if item not in self._parent:
            self._parent[item] = item

    def find(self, item: int) -> int:
        self._ensure(item)
        parent = self._parent[item]
        if parent != item:
            parent = self.find(parent)
            self._parent[item] = parent
        return parent

    def union(self, a: int, b: int) -> int:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return root_a
        # Always keep the smaller id as the canonical representative to retain determinism.
        if root_a < root_b:
            self._parent[root_b] = root_a
            return root_a
        self._parent[root_a] = root_b
        return root_b


class ComponentManager:
    """Tracks room and corridor connectivity using a disjoint set union."""

    def __init__(self) -> None:
        self._dsu = DisjointSetUnion()
        self._next_component_id = 0
        self._room_components: List[int] = []
        self._corridor_components: List[int] = []

    def new_component(self) -> int:
        component_id = self._next_component_id
        self._next_component_id += 1
        self._dsu._ensure(component_id)
        return component_id

    def _normalize(self, component_id: int) -> int:
        if component_id < 0:
            return -1
        return self._dsu.find(component_id)

    def register_room(self, component_id: int) -> int:
        root = self._normalize(component_id)
        self._room_components.append(root)
        return root

    def register_corridor(self, component_id: int) -> int:
        root = self._normalize(component_id)
        self._corridor_components.append(root)
        return root

    def set_room_component(self, index: int, component_id: int) -> int:
        root = self._normalize(component_id)
        self._room_components[index] = root
        return root

    def set_corridor_component(self, index: int, component_id: int) -> int:
        root = self._normalize(component_id)
        self._corridor_components[index] = root
        return root

    def room_component(self, index: int) -> int:
        root = self._normalize(self._room_components[index])
        self._room_components[index] = root
        return root

    def corridor_component(self, index: int) -> int:
        root = self._normalize(self._corridor_components[index])
        self._corridor_components[index] = root
        return root

    def normalize_component(self, component_id: int) -> int:
        return self._normalize(component_id)

    def union(self, *component_ids: int) -> int:
        valid_ids = [cid for cid in component_ids if cid >= 0]
        if not valid_ids:
            raise ValueError("Cannot merge empty component set")
        root = self._normalize(valid_ids[0])
        for cid in valid_ids[1:]:
            other_root = self._normalize(cid)
            root = self._dsu.union(root, other_root)
        return self._dsu.find(root)

    def components_equal(self, component_a: int, component_b: int) -> bool:
        if component_a < 0 or component_b < 0:
            return False
        return self._normalize(component_a) == self._normalize(component_b)

    def has_single_component(self) -> bool:
        return len(self._active_components()) <= 1

    def _active_components(self) -> Set[int]:
        components: Set[int] = set()
        for idx in range(len(self._room_components)):
            root = self.room_component(idx)
            if root >= 0:
                components.add(root)
        for idx in range(len(self._corridor_components)):
            root = self.corridor_component(idx)
            if root >= 0:
                components.add(root)
        return components

    def component_summary(self) -> Dict[int, Dict[str, List[int]]]:
        summary: Dict[int, Dict[str, List[int]]] = defaultdict(lambda: {"rooms": [], "corridors": []})
        for room_idx in range(len(self._room_components)):
            component_id = self.room_component(room_idx)
            summary[component_id]["rooms"].append(room_idx)
        for corridor_idx in range(len(self._corridor_components)):
            component_id = self.corridor_component(corridor_idx)
            summary[component_id]["corridors"].append(corridor_idx)
        return dict(summary)

    def component_sizes(self) -> Dict[int, int]:
        sizes = {}
        for room_idx in range(len(self._room_components)):
            component_id = self.room_component(room_idx)
            sizes[component_id] = sizes.get(component_id, 0) + 1
        for corridor_idx in range(len(self._corridor_components)):
            component_id = self.corridor_component(corridor_idx)
            sizes[component_id] = sizes.get(component_id, 0) + 1
        return sizes

    def total_components(self) -> int:
        return len(self._active_components())

    @property
    def room_count(self) -> int:
        return len(self._room_components)

    @property
    def corridor_count(self) -> int:
        return len(self._corridor_components)
