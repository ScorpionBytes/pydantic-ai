from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pydantic_graph.v2.id_types import NodeId


# TODO: Should StepContext be passed to joins/forks/decisions? Like, unified with ReducerContext etc.?
class StepContext[StateT, InputT]:
    """The main reason this is not a dataclass is that we need it to be covariant in its type parameters."""

    def __init__(self, state: StateT, inputs: InputT):
        self._state = state
        self._inputs = inputs

    @property
    def state(self) -> StateT:
        return self._state

    @property
    def inputs(self) -> InputT:
        return self._inputs

    def __repr__(self):
        return f'{self.__class__.__name__}(state={self.state}, inputs={self.inputs})'


class StepCallProtocol[StateT, InputT, OutputT](Protocol):
    """The purpose of this is to make it possible to deserialize step calls similar to how Evaluators work."""

    def __call__(self, ctx: StepContext[StateT, InputT]) -> OutputT:
        raise NotImplementedError


@dataclass
class Step[StateT, InputT, OutputT]:
    id: NodeId
    call: StepCallProtocol[StateT, InputT, OutputT]
