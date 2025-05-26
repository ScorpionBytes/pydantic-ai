from __future__ import annotations

from enum import Enum
from typing import Any

from typing_extensions import TypeGuard

from pydantic_graph.v2.decision import Decision
from pydantic_graph.v2.id_types import ForkId, NodeId
from pydantic_graph.v2.join import Join
from pydantic_graph.v2.spread import Spread
from pydantic_graph.v2.step import Step


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


START = StartNode.start
END = EndNode.end

type AnyMiddleNode = Step[Any, Any, Any, Any] | Join[Any, Any, Any, Any] | Spread[Any, Any, Any, Any]
type AnySourceNode = AnyMiddleNode | StartNode
type AnyDestinationNode = AnyMiddleNode | EndNode | Decision[Any, Any]
type AnyNode = AnySourceNode | AnyDestinationNode


def is_source(node: AnyNode) -> TypeGuard[AnySourceNode]:
    return isinstance(node, (StartNode, Step, Join))


def is_destination(node: AnyNode) -> TypeGuard[AnyDestinationNode]:
    return isinstance(node, (EndNode, Step, Join, Decision))


def get_default_spread_id(source: AnySourceNode, destination: AnyDestinationNode) -> ForkId:
    return ForkId(NodeId(f'__spread__:{source.id}:{destination.id}'))
