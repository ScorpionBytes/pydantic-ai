from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic_graph.v2.id_types import ForkId, JoinId, NodeRunId


class ReducerContext:
    pass


class Reducer[GraphStateT, InputT, OutputT]:
    def __init__(self, state: GraphStateT, inputs: InputT, downstream_fork_stack: tuple[tuple[ForkId, NodeRunId]]):
        self._state = state
        self._internal_state = None

    def cancel_other_requests(self) -> None:
        raise NotImplementedError

    def reduce(self, graph_state: GraphStateT, input: InputT) -> None:
        raise NotImplementedError

    def finalize(self, graph_state: GraphStateT) -> OutputT:
        raise NotImplementedError

    @staticmethod
    def list_reducer[T](item_type: type[T]) -> type[Reducer[object, T, list[T]]]:
        # append to list
        raise NotImplementedError

    @staticmethod
    def dict_reducer[T: dict[Any, Any]](
        dict_type: type[T],
    ) -> type[Reducer[object, T, T]]:
        # update dict
        raise NotImplementedError


type ReducerFactory[StateT, InputT, OutputT] = Callable[[StateT, InputT], Reducer[StateT, InputT, OutputT]]


@dataclass
class Join[StateT, InputT, OutputT]:
    id: JoinId

    reducer_factory: ReducerFactory[StateT, InputT, OutputT]

    # TODO: Need to implement a version of DominatingForkFinder that validates the specified NodeId is valid
    # Maybe should call this "parent_fork" or similar..
    joins: ForkId | None = None  # the NodeID of the node to use as the dominating fork
