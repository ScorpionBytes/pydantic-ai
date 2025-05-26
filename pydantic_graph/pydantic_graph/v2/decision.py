from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable, Never

from pydantic_graph.v2.id_types import NodeId
from pydantic_graph.v2.node import AnyDestinationNode, EndNode
from pydantic_graph.v2.transform import TransformFunction


@dataclass
class Decision[StateT, DepsT, SourceT, EndT]:
    id: NodeId
    branches: list[DecisionBranch[StateT, DepsT, Any, Any]]

    def with_branch[S, E, S2, E2](
        self: Decision[StateT, DepsT, S, E], branch: DecisionBranch[StateT, DepsT, S2, E2]
    ) -> Decision[StateT, DepsT, S | S2, E | E2]:
        return Decision(id=self.id, branches=self.branches + [branch])

    def _force_source_invariant(self, source: SourceT) -> SourceT:
        raise RuntimeError('This method should never be called, it is just defined for typing purposes.')

    def _force_end_covariant(self) -> EndT:
        raise RuntimeError('This method should never be called, it is just defined for typing purposes.')


@dataclass
class DecisionBranch[StateT, DepsT, SourceT, EndT]:
    source: type[SourceT]
    route_to: AnyDestinationNode
    # TODO: Rename `matches` to `test_match` or similar
    matches: Callable[[Any], bool] | None = None
    # TODO: If we change `transforms` to a single callable, we can make SourceT the type of the inputs
    transforms: tuple[TransformFunction[StateT, DepsT, Any, Any, Any], ...] = ()
    # TODO: the branch needs a node ID to use as the ID of the spread node
    spread: bool = False
    post_spread_transform: TransformFunction[StateT, DepsT, Any, Any, Any] | None = None


@dataclass
class DecisionBranchBuilder[StateT, DepsT, SourceT, EdgeInputT, EdgeOutputT]:
    source: type[SourceT]
    matches: Callable[[Any], bool] | None = None
    transforms: tuple[TransformFunction[StateT, DepsT, EdgeInputT, Any, Any], ...] = ()

    def transform[T](
        self,
        call: TransformFunction[StateT, DepsT, EdgeInputT, EdgeOutputT, T],
    ) -> DecisionBranchBuilder[StateT, DepsT, SourceT, EdgeInputT, T]:
        new_transforms = self.transforms + (call,)
        return DecisionBranchBuilder(self.source, self.matches, new_transforms)

    def route_to(  # analogous to GraphBuilder.edge
        self, node: AnyDestinationNode
    ) -> DecisionBranch[StateT, DepsT, SourceT, Never]:
        return DecisionBranch[StateT, DepsT, SourceT, Never](
            source=self.source,
            route_to=node,
            matches=self.matches,
            transforms=self.transforms,
        )

    def spread_to[T](  # analogous to GraphBuilder.spreading_edge
        self: DecisionBranchBuilder[StateT, DepsT, SourceT, EdgeInputT, Sequence[T]],
        node: AnyDestinationNode,
        post_spread_transform: TransformFunction[StateT, DepsT, Sequence[T], Any, Any] | None,
    ) -> DecisionBranch[StateT, DepsT, SourceT, Never]:
        return DecisionBranch[StateT, DepsT, SourceT, Never](
            source=self.source,
            route_to=node,
            matches=self.matches,
            transforms=self.transforms,
            spread=True,
            post_spread_transform=post_spread_transform,
        )

    def end(
        self,
    ) -> DecisionBranch[StateT, DepsT, SourceT, EdgeOutputT]:
        return DecisionBranch(
            source=self.source,
            route_to=EndNode.end,
            matches=self.matches,
            transforms=self.transforms,
        )
