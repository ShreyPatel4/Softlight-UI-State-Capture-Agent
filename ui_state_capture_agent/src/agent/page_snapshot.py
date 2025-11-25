from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SnapshotNode:
    index: int
    node_name: str
    attributes: dict[str, str]
    text_snippet: Optional[str] = None


@dataclass
class AXNode:
    node_id: str
    role: Optional[str]
    name: Optional[str]
    dom_node_indices: List[int] = field(default_factory=list)


@dataclass
class PageSnapshot:
    dom_nodes: List[SnapshotNode]
    ax_nodes: List[AXNode]
    by_dom_index: Dict[int, SnapshotNode] = field(default_factory=dict)
    ax_by_role: Dict[str, List[AXNode]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.by_dom_index:
            self.by_dom_index = {node.index: node for node in self.dom_nodes}
        if not self.ax_by_role:
            for ax in self.ax_nodes:
                if not ax.role:
                    continue
                self.ax_by_role.setdefault(ax.role.lower(), []).append(ax)

    @classmethod
    def from_nodes(cls, dom_nodes: List[SnapshotNode], ax_nodes: List[AXNode]) -> "PageSnapshot":
        return cls(dom_nodes=dom_nodes, ax_nodes=ax_nodes)
