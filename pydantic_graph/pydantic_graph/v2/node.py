from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pydantic_graph.v2.id_types import ForkId, NodeId


class StartNode(str, Enum):
    start = 'start'

    @property
    def id(self) -> NodeId:
        return NodeId(f'__{self.value}__')


class EndNode(str, Enum):
    end = 'end'

    @property
    def id(self) -> NodeId:
        return NodeId(f'__{self.value}__')


@dataclass
class Spread:
    id: ForkId


START = StartNode.start
END = EndNode.end
