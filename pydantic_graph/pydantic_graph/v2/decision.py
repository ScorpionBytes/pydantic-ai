from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable, Never

from pydantic_graph.v2.id_types import NodeId
from pydantic_graph.v2.node import AnyDestinationNode, EndNode
from pydantic_graph.v2.transform import TransformFunction


@dataclass
class Decision[SourceT, EndT]:
    id: NodeId
    branches: list[DecisionBranch[Any, Any]]

    def with_branch[S, E, S2, E2](self: Decision[S, E], branch: DecisionBranch[S2, E2]) -> Decision[S | S2, E | E2]:
        return Decision(id=self.id, branches=self.branches + [branch])

    def _force_source_invariant(self, source: SourceT) -> SourceT:
        raise RuntimeError('This method should never be called, it is just defined for typing purposes.')

    def _force_end_covariant(self) -> EndT:
        raise RuntimeError('This method should never be called, it is just defined for typing purposes.')


@dataclass
class DecisionBranch[SourceT, EndT]:
    source: type[SourceT]
    route_to: AnyDestinationNode
    # TODO: Rename `matches` to `test_match` or similar
    matches: Callable[[Any], bool] | None = None
    transforms: tuple[TransformFunction[Any, Any, Any, Any], ...] = ()
    # TODO: the branch needs a node ID to use as the ID of the spread node
    spread: bool = False
    post_spread_transform: TransformFunction[Any, Any, Any, Any] | None = None


@dataclass
class DecisionBranchBuilder[SourceT, GraphStateT, EdgeInputT, EdgeOutputT]:
    source: type[SourceT]
    matches: Callable[[Any], bool] | None = None
    transforms: tuple[TransformFunction[GraphStateT, EdgeInputT, Any, Any], ...] = ()

    def transform[T](
        self,
        call: TransformFunction[GraphStateT, EdgeInputT, EdgeOutputT, T],
    ) -> DecisionBranchBuilder[SourceT, GraphStateT, EdgeInputT, T]:
        new_transforms = self.transforms + (call,)
        return DecisionBranchBuilder(self.source, self.matches, new_transforms)

    def route_to(  # analogous to GraphBuilder.edge
        self, node: AnyDestinationNode
    ) -> DecisionBranch[SourceT, Never]:
        return DecisionBranch[SourceT, Never](
            source=self.source,
            route_to=node,
            matches=self.matches,
            transforms=self.transforms,
        )

    def spread_to[T](  # analogous to GraphBuilder.spreading_edge
        self: DecisionBranchBuilder[SourceT, GraphStateT, EdgeInputT, Sequence[T]],
        node: AnyDestinationNode,
        post_spread_transform: TransformFunction[GraphStateT, Sequence[T], Any, Any] | None,
    ) -> DecisionBranch[SourceT, Never]:
        return DecisionBranch[SourceT, Never](
            source=self.source,
            route_to=node,
            matches=self.matches,
            transforms=self.transforms,
            spread=True,
            post_spread_transform=post_spread_transform,
        )

    def end(
        self,
    ) -> DecisionBranch[SourceT, EdgeOutputT]:
        return DecisionBranch(
            source=self.source,
            route_to=EndNode.end,
            matches=self.matches,
            transforms=self.transforms,
        )


# TODO: Move this to being a GraphBuilder method that includes the graph state type
def handle[SourceT](
    case: type[SourceT], matches: Callable[[Any], bool] | None = None
) -> DecisionBranchBuilder[SourceT, Any, Any, Any]:
    if matches is None:
        if case in {Any, object}:

            def default_matches_branch(x: Any) -> bool:
                return True
        else:

            def default_matches_branch(x: Any) -> bool:
                return isinstance(x, case)

        matches = default_matches_branch

    return DecisionBranchBuilder(case, matches, transforms=())
