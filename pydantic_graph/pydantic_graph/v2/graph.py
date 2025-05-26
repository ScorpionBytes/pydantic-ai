from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Never, cast, get_args, get_origin, overload

from typing_extensions import assert_never

from pydantic_graph.v2.decision import Decision, DecisionBranchBuilder
from pydantic_graph.v2.id_types import ForkId, JoinId, NodeRunId
from pydantic_graph.v2.join import Join, Reducer, ReducerContext
from pydantic_graph.v2.mermaid import StateDiagramDirection, generate_code
from pydantic_graph.v2.node import (
    END,
    START,
    EndNode,
    NodeId,
    Spread,
    StartNode,
)
from pydantic_graph.v2.node_types import (
    AnyDestinationNode,
    AnyNode,
    AnySourceNode,
    get_default_spread_id,
    is_destination,
    is_source,
)
from pydantic_graph.v2.parent_forks import ParentFork, ParentForkFinder
from pydantic_graph.v2.step import Step, StepCallProtocol, StepContext
from pydantic_graph.v2.transform import AnyTransformFunction, TransformContext, TransformFunction
from pydantic_graph.v2.util import get_callable_name, get_unique_string


@dataclass
class Edge:
    source_id: NodeId
    transform: AnyTransformFunction | None
    destination_id: NodeId
    user_label: str | None

    def source(self, nodes: dict[NodeId, AnyNode]) -> AnySourceNode:
        node = nodes.get(self.source_id)
        if node is None:
            raise ValueError(f'Node {self.source_id} not found in graph')
        if not is_source(node):
            raise ValueError(f'Node {self.source_id} is not a source node: {node}')
        return node

    def destination(self, nodes: dict[NodeId, AnyNode]) -> AnyDestinationNode:
        node = nodes.get(self.destination_id)
        if node is None:
            raise ValueError(f'Node {self.destination_id} not found in graph')
        if not is_destination(node):
            raise ValueError(f'Node {self.destination_id} is not a source node: {node}')
        return node

    @property
    def label(self) -> str | None:
        # TODO: Add some default behavior?
        return self.user_label


class TypeUnion[T]:
    pass


@dataclass
class GraphBuilder[StateT, DepsT, GraphInputT, GraphOutputT]:
    state_type: type[StateT]
    deps_type: type[DepsT]
    input_type: type[GraphInputT]
    output_type: type[TypeUnion[GraphOutputT]] | type[GraphOutputT]

    _nodes: dict[NodeId, AnyNode] = field(init=False, default_factory=dict)
    _edges_by_source: dict[NodeId, list[Edge]] = field(init=False, default_factory=lambda: defaultdict(list))

    type Source[OutputT] = Step[StateT, DepsT, Any, OutputT] | Join[StateT, DepsT, Any, OutputT]
    type SourceWithInputs[InputT, OutputT] = Step[StateT, DepsT, InputT, OutputT] | Join[StateT, DepsT, InputT, OutputT]
    type Destination[InputT] = (
        Step[StateT, DepsT, InputT, Any]
        | Join[StateT, DepsT, InputT, Any]
        | Decision[StateT, DepsT, InputT, GraphOutputT]
    )

    # Node building:
    def build_step[InputT, OutputT](
        self,
        call: StepCallProtocol[StateT, DepsT, InputT, OutputT],
        *,
        node_id: str | None = None,
        label: str | None = None,
    ) -> Step[StateT, DepsT, InputT, OutputT]:
        if node_id is None:
            # TODO: Infer this from the parent frame variable assignment
            node_id = f'step-{get_callable_name(call)}-{get_unique_string()}'
        return Step[StateT, DepsT, InputT, OutputT](id=NodeId(node_id), call=call, user_label=label)

    def build_join[InputT, OutputT](
        self,
        reducer_factory: Callable[[StateT, DepsT, InputT], Reducer[StateT, DepsT, InputT, OutputT]],
        *,
        node_id: str | None = None,
    ) -> Join[StateT, DepsT, InputT, OutputT]:
        if node_id is None:
            # TODO: Infer this from the parent frame variable assignment
            node_id = f'join-{get_callable_name(reducer_factory)}-{get_unique_string()}'
        return Join[StateT, DepsT, InputT, OutputT](
            id=JoinId(NodeId(node_id)),
            reducer_factory=reducer_factory,
        )

    @staticmethod
    def decision(*, node_id: str | None = None) -> Decision[StateT, DepsT, Never, Never]:
        if node_id is None:
            node_id = f'decision-{get_unique_string()}'
        return Decision[StateT, DepsT, Never, Never](id=NodeId(node_id), branches=[])

    # TODO: Add a method more closely related to edge building that accepts the input node as a way to get a type-checked input
    #   Alternatively, add InputT as a type on Decision, and include it in the output of DecisionBranchBuilder, and do type-checking of it.
    def handle[SourceT](
        self, case: type[SourceT], *, matches: Callable[[Any], bool] | None = None, label: str | None = None
    ) -> DecisionBranchBuilder[StateT, DepsT, SourceT, Any, SourceT]:
        return DecisionBranchBuilder(case, matches, transforms=(), user_label=label)

    # note: forks are built by calls to `xyz_spread`, by calling `start_with` multiple times, or by calling `edge` multiple times with the same source

    # Edge building
    # Node "types" to be connected into edges: 'start', 'end', Step, Decision, Join, Fork.
    # You typically don't manually create forks — they are inferred from multiple edges coming out of a single node.
    @overload
    def start_with(
        self,
        destination: Destination[GraphInputT],
        *,
        label: str | None = None,
    ) -> None: ...

    @overload
    def start_with[DestinationInputT](
        self,
        destination: Destination[DestinationInputT],
        *,
        transform: TransformFunction[StateT, DepsT, GraphInputT, GraphInputT, DestinationInputT],
        label: str | None = None,
    ) -> None: ...

    def start_with(
        self,
        destination: Destination[Any],
        *,
        transform: TransformFunction[StateT, DepsT, Any, Any, Any] | None = None,
        label: str | None = None,
    ) -> None:
        self._add_edge_from_nodes(
            source=START,
            transform=transform,
            destination=destination,
            label=label,
        )

    @overload
    def start_with_spread[GraphInputItemT](
        self: GraphBuilder[StateT, DepsT, Sequence[GraphInputItemT], GraphOutputT],
        node: Destination[GraphInputItemT],
        *,
        pre_spread_label: str | None = None,
        post_spread_label: str | None = None,
    ) -> None: ...

    @overload
    def start_with_spread[DestinationInputT](
        self,
        node: Destination[DestinationInputT],
        *,
        pre_spread_transform: TransformFunction[StateT, DepsT, GraphInputT, GraphInputT, Sequence[DestinationInputT]],
        pre_spread_label: str | None = None,
        post_spread_label: str | None = None,
    ) -> None: ...

    @overload
    def start_with_spread[GraphInputItemT, DestinationInputT](
        self: GraphBuilder[StateT, DepsT, Sequence[GraphInputItemT], GraphOutputT],
        node: Destination[DestinationInputT],
        *,
        post_spread_transform: TransformFunction[
            StateT, DepsT, Sequence[GraphInputItemT], GraphInputItemT, DestinationInputT
        ],
        pre_spread_label: str | None = None,
        post_spread_label: str | None = None,
    ) -> None: ...

    @overload
    def start_with_spread[IntermediateT, DestinationInputT](
        self: GraphBuilder[StateT, DepsT, GraphInputT, GraphOutputT],
        node: Destination[DestinationInputT],
        *,
        pre_spread_transform: TransformFunction[StateT, DepsT, GraphInputT, GraphInputT, Sequence[IntermediateT]],
        post_spread_transform: TransformFunction[StateT, DepsT, GraphInputT, IntermediateT, DestinationInputT],
        pre_spread_label: str | None = None,
        post_spread_label: str | None = None,
    ) -> None: ...

    def start_with_spread(
        self,
        node: Destination[Any],
        *,
        pre_spread_transform: TransformFunction[StateT, DepsT, Any, Any, Sequence[Any]] | None = None,
        post_spread_transform: TransformFunction[StateT, DepsT, Any, Any, Any] | None = None,
        pre_spread_label: str | None = None,
        post_spread_label: str | None = None,
    ) -> None:
        # TODO: Need to allow specifying the id manually to prevent conflicts if there are multiple spreads between the same two nodes
        spread = Spread(id=get_default_spread_id(START, node))
        self._add_edge_from_nodes(
            source=START,
            transform=pre_spread_transform,
            destination=spread,
            label=pre_spread_label,
        )
        self._add_edge_from_nodes(
            source=spread,
            transform=post_spread_transform,
            destination=node,
            label=post_spread_label,
        )

    @overload
    def edge[SourceOutputT](
        self,
        source: Source[SourceOutputT],
        destination: Destination[SourceOutputT],
        *,
        label: str | None = None,
    ) -> None: ...

    @overload
    def edge[SourceInputT, SourceOutputT, DestinationInputT](
        self,
        source: SourceWithInputs[SourceInputT, SourceOutputT],
        destination: Destination[DestinationInputT],
        *,
        transform: TransformFunction[StateT, DepsT, SourceInputT, SourceOutputT, DestinationInputT],
        label: str | None = None,
    ) -> None: ...

    def edge(
        self,
        source: Source[Any],
        destination: Destination[Any],
        *,
        transform: AnyTransformFunction | None = None,
        label: str | None = None,
    ) -> None:
        self._add_edge_from_nodes(
            source=source,
            transform=transform,
            destination=destination,
            label=label,
        )

    @overload
    def spreading_edge[SourceInputT, DestinationInputT](
        self,
        source: SourceWithInputs[SourceInputT, Sequence[DestinationInputT]],
        destination: Destination[DestinationInputT],
        *,
        pre_spread_label: str | None = None,
        post_spread_label: str | None = None,
    ) -> None: ...

    @overload
    def spreading_edge[SourceInputT, SourceOutputT, DestinationInputT](
        self,
        source: SourceWithInputs[SourceInputT, SourceOutputT],
        destination: Destination[DestinationInputT],
        *,
        pre_spread_transform: TransformFunction[
            StateT, DepsT, SourceInputT, SourceOutputT, Sequence[DestinationInputT]
        ],
        pre_spread_label: str | None = None,
        post_spread_label: str | None = None,
    ) -> None: ...

    @overload
    def spreading_edge[SourceInputT, SourceOutputItemT, DestinationInputT](
        self,
        source: SourceWithInputs[SourceInputT, Sequence[SourceOutputItemT]],
        destination: Destination[DestinationInputT],
        *,
        post_spread_transform: TransformFunction[
            StateT,
            DepsT,
            SourceInputT,
            SourceOutputItemT,
            DestinationInputT,
        ],
        pre_spread_label: str | None = None,
        post_spread_label: str | None = None,
    ) -> None: ...

    @overload
    def spreading_edge[SourceInputT, SourceOutputT, IntermediateT, DestinationInputT](
        self,
        source: SourceWithInputs[SourceInputT, SourceOutputT],
        destination: Destination[DestinationInputT],
        *,
        pre_spread_transform: TransformFunction[StateT, DepsT, SourceInputT, SourceOutputT, Sequence[IntermediateT]],
        post_spread_transform: TransformFunction[StateT, DepsT, SourceInputT, IntermediateT, DestinationInputT],
        pre_spread_label: str | None = None,
        post_spread_label: str | None = None,
    ) -> None: ...

    def spreading_edge[SourceInputT](
        self,
        source: SourceWithInputs[SourceInputT, Any],
        destination: Destination[Any],
        *,
        pre_spread_transform: TransformFunction[StateT, DepsT, SourceInputT, Any, Sequence[Any]] | None = None,
        post_spread_transform: TransformFunction[StateT, DepsT, SourceInputT, Any, Any] | None = None,
        pre_spread_label: str | None = None,
        post_spread_label: str | None = None,
    ) -> None:
        # TODO: Need to allow specifying the id manually to prevent conflicts if there are multiple spreads between the same two nodes
        fork = Spread(id=get_default_spread_id(source, destination))
        self._add_edge_from_nodes(
            source=source,
            transform=pre_spread_transform,
            destination=fork,
            label=pre_spread_label,
        )
        self._add_edge_from_nodes(
            source=fork,
            transform=post_spread_transform,
            destination=destination,
            label=post_spread_label,
        )

    @overload
    def end_from(
        self,
        source: Source[GraphOutputT],
        *,
        label: str | None = None,
    ) -> None: ...

    @overload
    def end_from[SourceInputT, SourceOutputT](
        self,
        source: SourceWithInputs[SourceInputT, SourceOutputT],
        *,
        label: str | None = None,
        transform: TransformFunction[StateT, DepsT, SourceInputT, SourceOutputT, GraphOutputT],
    ) -> None: ...

    def end_from(
        self,
        source: Source[Any],
        *,
        label: str | None = None,
        transform: TransformFunction[StateT, DepsT, Any, Any, GraphOutputT] | None = None,
    ) -> None:
        self._add_edge_from_nodes(
            source=source,
            transform=transform,
            destination=END,
            label=label,
        )

    def _add_edge_from_nodes(
        self,
        *,
        source: AnySourceNode,
        transform: AnyTransformFunction | None,
        destination: AnyDestinationNode,
        label: str | None = None,
    ) -> None:
        self._add_node(source)
        self._add_node(destination)

        edge = Edge(source_id=source.id, transform=transform, destination_id=destination.id, user_label=label)
        self._add_edge(edge)

    def _add_node(self, node: AnyNode) -> None:
        existing = self._nodes.get(node.id)
        if existing is None or isinstance(existing, (StartNode, EndNode)):
            pass  # it's not a problem to have non-unique instances of StartNode and EndNode
        elif existing is not node:
            raise ValueError(f'All nodes must have unique node IDs. {node.id!r} was the ID for {existing} and {node}')

        self._nodes[node.id] = node

    def _add_edge(self, edge: Edge) -> None:
        assert edge.source_id in self._nodes, f'Edge source {edge.source_id} not found in graph'
        assert edge.destination_id in self._nodes, f'Edge destination {edge.destination_id} not found in graph'
        self._edges_by_source[edge.source_id].append(edge)

    def build(self) -> Graph[StateT, DepsT, GraphInputT, GraphOutputT]:
        # TODO: Warn/error if the graph is not connected
        # TODO: Warn/error if any non-End node is a dead end
        # TODO: Error if the graph does not meet the every-join-has-a-source-fork requirement (otherwise can't know when to proceed past joins)
        # TODO: Convert decisions with spreads into "normal" spreads
        # TODO: Allow the user to specify the dominating forks; only infer them if _not_ specified
        # TODO: Verify that any user-specified dominating nodes are _actually_ dominating forks, and if not, generate a helpful error message
        # TODO: Consider doing a deepcopy here to prevent modifications to the underlying nodes and edges
        nodes = self._nodes
        edges = self._edges_by_source
        nodes, edges = _convert_decision_spreads(nodes, edges)
        parent_forks = _collect_dominating_forks(nodes, edges)

        output_type = cast(type[GraphOutputT], self.output_type)
        if get_origin(output_type) is TypeUnion:
            output_type = get_args(output_type)[0]

        return Graph[StateT, DepsT, GraphInputT, GraphOutputT](
            state_type=self.state_type,
            deps_type=self.deps_type,
            input_type=self.input_type,
            output_type=output_type,
            nodes=self._nodes,
            edges_by_source=self._edges_by_source,
            parent_forks=parent_forks,
        )


def _convert_decision_spreads(
    graph_nodes: dict[NodeId, AnyNode], graph_edges_by_source: dict[NodeId, list[Edge]]
) -> tuple[dict[NodeId, AnyNode], dict[NodeId, list[Edge]]]:
    # nodes = dict(graph_nodes)
    # edges: dict[NodeId, list[Edge]] = defaultdict(list)
    # edges.update(graph_edges_by_source)
    # TODO: Decide whether to do mutating updates in this function or not...
    nodes = graph_nodes
    edges = graph_edges_by_source

    for node in list(nodes.values()):
        if isinstance(node, Decision):
            for branch in node.branches:
                if branch.spread:
                    spread = Spread(id=ForkId(NodeId(f'spread:{branch.route_to.id}')))
                    old_route_to = branch.route_to
                    nodes[spread.id] = spread
                    edges[spread.id].append(
                        Edge(
                            source_id=spread.id,
                            transform=branch.post_spread_transform,
                            destination_id=old_route_to.id,
                            user_label=branch.post_spread_user_label,
                        )
                    )
                    branch.route_to = spread
                    branch.spread = False
                    branch.post_spread_transform = None
    return nodes, edges


def _collect_dominating_forks(
    graph_nodes: dict[NodeId, AnyNode], graph_edges_by_source: dict[NodeId, list[Edge]]
) -> dict[JoinId, ParentFork[NodeId]]:
    nodes = set(graph_nodes)
    start_ids = {StartNode.start.id}
    edges = {source_id: [e.destination_id for e in graph_edges_by_source[source_id]] for source_id in nodes}
    fork_ids = {
        node_id for node_id, node in graph_nodes.items() if isinstance(node, Spread) or len(edges.get(node_id, [])) > 1
    }
    finder = ParentForkFinder(
        nodes=nodes,
        start_ids=start_ids,
        fork_ids=fork_ids,
        edges=edges,
    )

    join_ids = {node.id for node in graph_nodes.values() if isinstance(node, Join)}
    dominating_forks: dict[JoinId, ParentFork[NodeId]] = {}
    for join_id in join_ids:
        dominating_fork = finder.find_parent_fork(join_id)
        if dominating_fork is None:
            # TODO: Print out the mermaid graph and explain the problem
            raise ValueError(f'Join node {join_id} has no dominating fork')
        dominating_forks[join_id] = dominating_fork

    return dominating_forks


@dataclass(repr=False)
class Graph[StateT, DepsT, InputT, OutputT]:
    state_type: type[StateT]
    deps_type: type[DepsT]
    input_type: type[InputT]
    output_type: type[OutputT]

    nodes: dict[NodeId, AnyNode]
    edges_by_source: dict[NodeId, list[Edge]]
    parent_forks: dict[JoinId, ParentFork[NodeId]]

    @property
    def start_edges(self) -> list[Edge]:
        return self.edges_by_source.get(START.id, [])

    def get_parent_fork(self, join_id: JoinId) -> ParentFork[NodeId]:
        result = self.parent_forks.get(join_id)
        if result is None:
            raise RuntimeError(f'Node {join_id} is not a join node or did not have a dominating fork (this is a bug)')
        return result

    async def run(self, state: StateT, deps: DepsT, inputs: InputT) -> OutputT:
        return await GraphRun(self, state, deps, inputs).run()

    def render(self, *, title: str | None = None, direction: StateDiagramDirection | None = None) -> str:
        return generate_code(self, title=title, direction=direction)

    def __repr__(self):
        return self.render()


@dataclass
class GraphWalkState:
    # id: WalkerId

    # With our current BaseNode thing, next_node_id and next_node_inputs are merged into `next_node` itself
    node_id: NodeId
    context_inputs: Any
    """
    usually this is the same as node_inputs, but it is different when working with spreads
    """
    node_inputs: Any
    fork_stack: tuple[tuple[ForkId, NodeRunId], ...]
    """
    Stack of forks that have been entered; used so that the GraphRunner can decide when to proceed through joins
    """


# TODO: Move `Some` and `Maybe` to util?
@dataclass
class Some[T]:
    value: T


type Maybe[T] = Some[T] | None  # like optional, but you can tell the difference between "no value" and "value is None"


@dataclass(init=False)
class GraphRun[StateT, DepsT, InputT, OutputT]:
    graph: Graph[StateT, DepsT, InputT, OutputT]
    state: StateT
    deps: DepsT
    inputs: InputT

    # persistence: Any  # TODO: Implement use of this

    def __init__(self, graph: Graph[StateT, DepsT, InputT, OutputT], state: StateT, deps: DepsT, inputs: InputT):
        self.graph = graph
        self.state = state
        self.deps = deps
        self.inputs = inputs

        self.result: Maybe[OutputT] = None
        self.active_walks: list[GraphWalkState] = [
            GraphWalkState(node_id=START.id, context_inputs=self.inputs, node_inputs=self.inputs, fork_stack=())
        ]

        self.active_reducers: dict[tuple[NodeId, NodeRunId], Reducer[StateT, DepsT, Any, Any]] = {}
        """The node id is for the join, the node run id is for the dominating fork."""

    async def run(self):
        # TODO: Refactor this to actually run distinct walks in parallel in the async event loop using a task group
        #   I'm implementing it in a blocking way for now to get the basic functionality working
        while self.active_walks:
            walk = self.active_walks.pop()
            node = self.graph.nodes[walk.node_id]

            if isinstance(node, StartNode):
                self._handle_start(walk)
            elif isinstance(node, Step):
                self._handle_step(node, walk)
            elif isinstance(node, Join):
                self._handle_reduce_join(node, walk)
            elif isinstance(node, Spread):
                self._handle_spread(walk)
            elif isinstance(node, Decision):
                self._handle_decision(node, walk)
            elif isinstance(node, EndNode):
                self._handle_end(walk)

            # Now that we've handled edges for the node, we can check if any joins are ready to proceed, and if so, proceed
            self._handle_finalize_joins(walk)

        if self.result is None:
            raise RuntimeError(
                'Graph run completed, but no result was produced. This is either a bug in the graph or a bug in the graph runner.'
            )

        return self.result.value

    def _handle_start(self, walk: GraphWalkState) -> None:
        # nothing to do besides start the graph
        self._handle_edges(walk, walk.context_inputs, walk.node_inputs)

    def _handle_step(self, step: Step[Any, Any, Any, Any], walk: GraphWalkState):
        step_context = StepContext(self.state, self.deps, walk.context_inputs)
        output = step.call(step_context)
        self._handle_edges(walk, output, output)

    def _handle_reduce_join(self, join: Join[Any, Any, Any, Any], walk: GraphWalkState) -> None:
        # Find the matching fork run id in the stack; this will be used to look for an active reducer
        parent_fork = self.graph.get_parent_fork(join.id)
        matching_fork_run_id = next(iter(x[1] for x in walk.fork_stack[::-1] if x[0] == parent_fork.fork_id), None)
        if matching_fork_run_id is None:
            raise RuntimeError(
                f'Fork {parent_fork.fork_id} not found in stack {walk.fork_stack}. This means the dominating fork is not dominating (this is a bug).'
            )

        # Get or create the active reducer
        reducer = self.active_reducers.get((join.id, matching_fork_run_id))
        if reducer is None:
            reducer = join.reducer_factory(self.state, self.deps, walk.node_inputs)
            self.active_reducers[(join.id, matching_fork_run_id)] = reducer

        # Reduce
        ctx = ReducerContext(self.state, self.deps, walk.node_inputs)
        reducer.reduce(ctx)

    def _handle_spread(self, walk: GraphWalkState):
        self._handle_edges(walk, walk.context_inputs, walk.node_inputs)

    def _handle_decision(self, decision: Decision[StateT, DepsT, Any, Any], walk: GraphWalkState) -> None:
        for branch in decision.branches:
            assert not branch.spread, 'Spreads decisions should be converted into spreads as part of graph-building'

            match_tester = branch.matches
            if match_tester is not None:
                inputs_match = match_tester(walk.node_inputs)
            elif branch.source in {Any, object}:
                inputs_match = True
            else:
                inputs_match = isinstance(walk.node_inputs, branch.source)

            if inputs_match:
                node_inputs = walk.node_inputs
                for transform in branch.transforms:
                    ctx = TransformContext(self.state, self.deps, walk.context_inputs, node_inputs)
                    node_inputs = transform(ctx)
                self.active_walks.append(
                    GraphWalkState(branch.route_to.id, walk.context_inputs, walk.node_inputs, walk.fork_stack)
                )
                break

    def _handle_end(self, walk: GraphWalkState) -> None:
        self.result = Some(walk.node_inputs)
        # TODO: Probably want to cancel all other walks, terminate the run, etc.

    def _handle_finalize_joins(self, popped_walk: GraphWalkState) -> None:
        # If the popped walk was the last item preventing one or more joins, those joins can now be finalized
        walk_fork_run_ids = {fork_run_id: i for i, (_, fork_run_id) in enumerate(popped_walk.fork_stack)}
        active_reducers_items = list(
            self.active_reducers.items()
        )  # make a copy to avoid modifying the dict while iterating

        # Note: might be more efficient to maintain a better data structure for looking up reducers by join_id and
        # fork_run_id without iterating through every item. This only matters if there is a large number of reducers.
        for (join_id, fork_run_id), reducer in active_reducers_items:
            fork_run_index = walk_fork_run_ids.get(fork_run_id)
            if fork_run_index is not None:
                # This reducer _may_ now be ready to finalize:
                join_can_proceed = True
                for walk in self.active_walks:
                    # might be a good idea to hold walks_by_fork_id in memory to reduce overhead here
                    if fork_run_id in {x[1] for x in walk.fork_stack}:
                        join_can_proceed = False

                if join_can_proceed:
                    ctx = ReducerContext(self.state, self.deps, None)
                    output = reducer.finalize(ctx)
                    new_fork_stack = popped_walk.fork_stack[:fork_run_index]
                    self.active_reducers.pop((join_id, fork_run_id))
                    # Should _now_ traverse the edges leaving this join
                    self._handle_edges(GraphWalkState(join_id, None, None, new_fork_stack), output, output)

    def _handle_edges(self, walk: GraphWalkState, context_inputs: Any, next_node_inputs: Any) -> None:
        edges = self.graph.edges_by_source.get(walk.node_id, [])
        node = self.graph.nodes[walk.node_id]

        fork_stack = walk.fork_stack
        if len(edges) > 1 or isinstance(node, Spread):
            # first condition is a broadcast fork; note that the way graph building works,
            # spread nodes should never be broadcast edges; even if there are multiple spreads between two nodes
            # these should result in distinct spread instances
            node_run_id = NodeRunId(str(uuid.uuid4()))
            fork_stack += ((ForkId(walk.node_id), node_run_id),)

        assert not isinstance(node, (Decision, EndNode)), 'This method should not be called for Decision, EndNode'
        if isinstance(node, (StartNode, Step, Join)):
            # Edge transitions should be fast, so maybe don't need to be handled in parallel
            for edge in edges:
                if edge.transform is not None:
                    transform_context = TransformContext(self.state, self.deps, context_inputs, next_node_inputs)
                    next_node_inputs = edge.transform(transform_context)

                self.active_walks.append(
                    GraphWalkState(edge.destination_id, context_inputs, next_node_inputs, fork_stack)
                )
        elif isinstance(node, Spread):
            for edge in edges:
                for item in next_node_inputs:
                    if edge.transform is not None:
                        transform_context = TransformContext(self.state, self.deps, context_inputs, item)
                        item = edge.transform(transform_context)
                    self.active_walks.append(GraphWalkState(edge.destination_id, context_inputs, item, fork_stack))
        else:
            assert_never(node)

    # async def run(self):
    #     from anyio import create_task_group, create_memory_object_stream
    #     from anyio.streams.memory import MemoryObjectReceiveStream
    #     send_result_stream, receive_result_stream = create_memory_object_stream[Any]()
    #
    #     async def run_step(step_ref: GraphWalkStep, inputs: Any):
    #         # # TODO: Handle Spread, Decision, StartNode, Step, Join, EndNode
    #         node = self.graph.nodes[step_ref.node_id]
    #         output = inputs
    #         if isinstance(node, Step):
    #             step_context = StepContext(self.state, inputs)
    #             output = node.call(step_context)
    #         elif isinstance(node, Join):
    #             dominating_fork = self.graph.get_dominating_fork(node.id)
    #             matching_fork = next(iter((x for x in step_ref.fork_stack[::-1] if x[0] == dominating_fork.fork_id)),
    #                                  None)
    #             if matching_fork is None:
    #                 raise RuntimeError(
    #                     f'Fork {node.id} not found in stack {step_ref.fork_stack}. This means the dominating fork is not dominating (this is a bug).')
    #             fork_node_run_id = matching_fork[1]
    #             active_reducer = self.active_reducers.get((node.id, fork_node_run_id))
    #             if active_reducer is None:
    #                 active_reducer = node.reducer_factory(self.state, inputs)
    #                 self.active_reducers[(node.id, fork_node_run_id)] = active_reducer
    #             active_reducer.reduce(self.state, inputs)
    #         elif isinstance(node, Spread):
    #             # TODO: The API currently suggests that you could access the output of the previous (pre-Spread) step, but it doesn't work that way now.
    #             pass
    #
    #         # TODO: Remove the reference to this task so that the handle_edges can check if any joins should proceed...
    #         await handle_edges(node, inputs, output)
    #
    #     async def handle_edges(source: AnyNode, inputs: Any, outputs: Any) -> None:
    #         edges = self.graph.edges_by_source.get(source.id, [])
    #         # Edge transitions should be fast, so don't need to be handled in parallel
    #         for edge in edges:
    #             next_steps = ...
    #             destination = edge.destination(self.graph.nodes)
    #             if isinstance(destination, EndNode):
    #         if isinstance(source, StartNode):
    #
    #
    #         # assert not isinstance(node, (EndNode, Spread, Decision))  # should be Start, Step, Join
    #         # if isinstance(node, StartNode):
    #         #     for edge in self.graph.edges_by_source[step.node_id]:
    #         #         # TODO: Should transforms be done in parallel?
    #         #         if edge.transform
    #         # step_result = step.run(self.graph, self.state)
    #         # await send_result_stream.send(step_result)
    #
    #     async def process_results(receive_stream: MemoryObjectReceiveStream[NodeExecutionResult]) -> None:
    #         async with receive_stream:
    #             async for result in receive_stream:
    #                 should_end_run = await self.handle_node_result(result)
    #                 if should_end_run:
    #                     # Exit any running tasks; useful for eager exit
    #                     tg.cancel_scope.cancel()
    #
    #     async with create_task_group() as tg:
    #         async with send_result_stream:
    #             tg.start_soon(process_results, receive_result_stream)
    #
    #             first_step = GraphWalkStep(
    #                 node_id=START.id,
    #                 inputs=self.inputs,
    #                 fork_stack=[],
    #             )
    #             tg.start_soon(run_step, first_step)

    # async def handle_node_result(self, result: NodeExecutionResult) -> bool:
    #     # Returns True if the full run is complete, so that any remaining-but-no-longer-relevant tasks can be canceled
    #     # TODO: Implement..
    #     raise NotImplementedError
