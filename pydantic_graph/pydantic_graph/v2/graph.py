from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Never, overload

from typing_extensions import assert_never

from pydantic_graph.v2.decision import Decision
from pydantic_graph.v2.dominating_forks import DominatingFork, DominatingForkFinder
from pydantic_graph.v2.id_types import ForkId, JoinId, NodeRunId
from pydantic_graph.v2.join import Join, Reducer
from pydantic_graph.v2.node import (
    END,
    START,
    AnyDestinationNode,
    AnyNode,
    AnySourceNode,
    EndNode,
    NodeId,
    StartNode,
    get_default_spread_id,
    is_destination,
    is_source,
)
from pydantic_graph.v2.spread import Spread
from pydantic_graph.v2.step import Step, StepCallProtocol, StepContext
from pydantic_graph.v2.transform import TransformContext, TransformFunction
from pydantic_graph.v2.util import get_callable_name, get_unique_string


@dataclass
class Edge:
    source_id: NodeId
    transform: TransformFunction[Any, Any, Any, Any] | None
    destination_id: NodeId

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


class GraphBuilder[StateT, GraphInputT, GraphOutputT]:
    state_type: type[StateT]
    input_type: type[GraphInputT]
    output_type: type[GraphOutputT]

    _nodes: dict[NodeId, AnyNode]
    _edges_by_source: dict[NodeId, list[Edge]]
    _edges_by_destination: dict[NodeId, list[Edge]]

    type Source[OutputT] = Step[StateT, Any, OutputT] | Join[StateT, Any, OutputT]
    type SourceWithInputs[InputT, OutputT] = Step[StateT, InputT, OutputT] | Join[StateT, InputT, OutputT]
    type Destination[InputT] = Step[StateT, InputT, Any] | Join[StateT, InputT, Any] | Decision[InputT, GraphOutputT]

    # Node building:
    def build_step[InputT, OutputT](
        self, call: StepCallProtocol[StateT, InputT, OutputT]
    ) -> Step[StateT, InputT, OutputT]:
        return Step[StateT, InputT, OutputT](
            id=NodeId(f'step-{get_callable_name(call)}-{get_unique_string()}'), call=call
        )

    def build_join[InputT, OutputT](
        self, reducer_factory: Callable[[StateT, InputT], Reducer[StateT, InputT, OutputT]]
    ) -> Join[StateT, InputT, OutputT]:
        # TODO: Need to allow specifying the id manually to prevent conflicts if there are multiple spreads between the same two nodes
        return Join[StateT, InputT, OutputT](
            id=JoinId(NodeId(f'join-{get_callable_name(reducer_factory)}-{get_unique_string()}')),
            reducer_factory=reducer_factory,
        )

    @staticmethod
    def decision(id: str | None) -> Decision[Never, Never]:
        # TODO: Generate a deterministic node ID and/or require it be provided.
        #   It's probably enough to generate a name as a function of how many nodes have been created,
        #   noting that if you want repeatability you should always specify the node ID
        return Decision[Never, Never](id=NodeId(id or '-'), branches=[])

    # note: forks are built by calls to `xyz_spread`, by calling `start_with` multiple times, or by calling `edge` multiple times with the same source

    # TODO: Need to add `handle` method for DecisionBranch somewhere that is aware of the graph state, possibly here

    # Edge building
    # Node "types" to be connected into edges: 'start', 'end', Step, Decision, Join, Fork.
    # You typically don't manually create forks — they are inferred from multiple edges coming out of a single node.
    @overload
    def start_with(self, destination: Destination[GraphInputT]) -> None: ...

    @overload
    def start_with[DestinationInputT](
        self,
        destination: Destination[DestinationInputT],
        *,
        transform: TransformFunction[StateT, GraphInputT, GraphInputT, DestinationInputT],
    ) -> None: ...

    def start_with(
        self,
        destination: Destination[Any],
        *,
        transform: TransformFunction[StateT, Any, Any, Any] | None = None,
    ) -> None:
        self._add_edge_from_nodes(
            source=START,
            transform=transform,
            destination=destination,
        )

    @overload
    def start_with_spread[GraphInputItemT](
        self: GraphBuilder[StateT, Sequence[GraphInputItemT], GraphOutputT],
        node: Destination[GraphInputItemT],
    ) -> None: ...

    @overload
    def start_with_spread[DestinationInputT](
        self,
        node: Destination[DestinationInputT],
        *,
        pre_spread_transform: TransformFunction[StateT, GraphInputT, GraphInputT, Sequence[DestinationInputT]],
    ) -> None: ...

    @overload
    def start_with_spread[GraphInputItemT, DestinationInputT](
        self: GraphBuilder[StateT, Sequence[GraphInputItemT], GraphOutputT],
        node: Destination[DestinationInputT],
        *,
        post_spread_transform: TransformFunction[StateT, Sequence[GraphInputItemT], GraphInputItemT, DestinationInputT],
    ) -> None: ...

    @overload
    def start_with_spread[IntermediateT, DestinationInputT](
        self: GraphBuilder[StateT, GraphInputT, GraphOutputT],
        node: Destination[DestinationInputT],
        *,
        pre_spread_transform: TransformFunction[StateT, GraphInputT, GraphInputT, Sequence[IntermediateT]],
        post_spread_transform: TransformFunction[StateT, GraphInputT, IntermediateT, DestinationInputT],
    ) -> None: ...

    def start_with_spread(
        self,
        node: Destination[Any],
        *,
        pre_spread_transform: TransformFunction[StateT, Any, Any, Sequence[Any]] | None = None,
        post_spread_transform: TransformFunction[StateT, Any, Any, Any] | None = None,
    ) -> None:
        # TODO: Need to allow specifying the id manually to prevent conflicts if there are multiple spreads between the same two nodes
        spread = Spread[Any, Any, Any](id=get_default_spread_id(START, node))
        self._add_edge_from_nodes(
            source=START,
            transform=pre_spread_transform,
            destination=spread,
        )
        self._add_edge_from_nodes(
            source=spread,
            transform=post_spread_transform,
            destination=node,
        )

    @overload
    def edge[SourceOutputT](self, source: Source[SourceOutputT], destination: Destination[SourceOutputT]) -> None: ...

    @overload
    def edge[SourceInputT, SourceOutputT, DestinationInputT](
        self,
        source: SourceWithInputs[SourceInputT, SourceOutputT],
        destination: Destination[DestinationInputT],
        *,
        transform: TransformFunction[StateT, SourceInputT, SourceOutputT, DestinationInputT],
    ) -> None: ...

    def edge(
        self,
        source: Source[Any],
        destination: Destination[Any],
        *,
        transform: TransformFunction[Any, Any, Any, Any] | None = None,
    ) -> None:
        self._add_edge_from_nodes(
            source=source,
            transform=transform,
            destination=destination,
        )

    @overload
    def spreading_edge[SourceInputT, DestinationInputT](
        self,
        source: SourceWithInputs[SourceInputT, Sequence[DestinationInputT]],
        destination: Destination[DestinationInputT],
    ) -> None: ...

    @overload
    def spreading_edge[SourceInputT, SourceOutputT, DestinationInputT](
        self,
        source: SourceWithInputs[SourceInputT, SourceOutputT],
        destination: Destination[DestinationInputT],
        *,
        pre_spread_transform: TransformFunction[StateT, SourceInputT, SourceOutputT, Sequence[DestinationInputT]],
    ) -> None: ...

    @overload
    def spreading_edge[SourceInputT, SourceOutputItemT, DestinationInputT](
        self,
        source: SourceWithInputs[SourceInputT, Sequence[SourceOutputItemT]],
        destination: Destination[DestinationInputT],
        *,
        post_spread_transform: TransformFunction[
            StateT,
            SourceInputT,
            SourceOutputItemT,
            DestinationInputT,
        ],
    ) -> None: ...

    @overload
    def spreading_edge[SourceInputT, SourceOutputT, IntermediateT, DestinationInputT](
        self,
        source: SourceWithInputs[SourceInputT, SourceOutputT],
        destination: Destination[DestinationInputT],
        *,
        pre_spread_transform: TransformFunction[StateT, SourceInputT, SourceOutputT, Sequence[IntermediateT]],
        post_spread_transform: TransformFunction[StateT, SourceInputT, IntermediateT, DestinationInputT],
    ) -> None: ...

    def spreading_edge[SourceInputT](
        self,
        source: SourceWithInputs[SourceInputT, Any],
        destination: Destination[Any],
        *,
        pre_spread_transform: TransformFunction[StateT, SourceInputT, Any, Sequence[Any]] | None = None,
        post_spread_transform: TransformFunction[StateT, SourceInputT, Any, Any] | None = None,
    ) -> None:
        # TODO: Need to allow specifying the id manually to prevent conflicts if there are multiple spreads between the same two nodes
        fork = Spread[Any, Any, Any](
            id=get_default_spread_id(source, destination),
        )
        self._add_edge_from_nodes(
            source=source,
            transform=pre_spread_transform,
            destination=fork,
        )
        self._add_edge_from_nodes(
            source=fork,
            transform=post_spread_transform,
            destination=destination,
        )

    @overload
    def end_from(self, source: Source[GraphOutputT]) -> None: ...

    @overload
    def end_from[SourceInputT, SourceOutputT](
        self,
        source: SourceWithInputs[SourceInputT, SourceOutputT],
        *,
        transform: TransformFunction[StateT, SourceInputT, SourceOutputT, GraphOutputT],
    ) -> None: ...

    def end_from(
        self,
        source: Source[Any],
        *,
        transform: TransformFunction[StateT, Any, Any, GraphOutputT] | None = None,
    ) -> None:
        self._add_edge_from_nodes(
            source=source,
            transform=transform,
            destination=END,
        )

    def _add_edge_from_nodes(
        self,
        *,
        source: AnySourceNode,
        transform: TransformFunction[Any, Any, Any, Any] | None,
        destination: AnyDestinationNode,
    ) -> None:
        self._add_node(source)
        self._add_node(destination)

        edge = Edge(source_id=source.id, transform=transform, destination_id=destination.id)
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
        self._edges_by_destination[edge.destination_id].append(edge)

    def build(self) -> Graph[StateT, GraphInputT, GraphOutputT]:
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
        dominating_forks = _collect_dominating_forks(nodes, edges)

        return Graph[StateT, GraphInputT, GraphOutputT](
            # deps_type=self.deps_type,
            state_type=self.state_type,
            input_type=self.input_type,
            output_type=self.output_type,
            nodes=self._nodes,
            edges_by_source=self._edges_by_source,
            dominating_forks=dominating_forks,
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
                    spread = Spread[Any, Any, Any](id=ForkId(NodeId(f'spread:{branch.route_to.id}')))
                    old_route_to = branch.route_to
                    nodes[spread.id] = spread
                    edges[spread.id].append(
                        Edge(
                            source_id=spread.id,
                            transform=branch.post_spread_transform,
                            destination_id=old_route_to.id,
                        )
                    )
                    branch.route_to = spread
                    branch.spread = False
                    branch.post_spread_transform = None
    return nodes, edges


def _collect_dominating_forks(
    graph_nodes: dict[NodeId, AnyNode], graph_edges_by_source: dict[NodeId, list[Edge]]
) -> dict[JoinId, DominatingFork[NodeId]]:
    nodes = set(graph_nodes)
    start_ids = {StartNode.start.id}
    edges = {source_id: [e.destination_id for e in graph_edges_by_source[source_id]] for source_id in nodes}
    fork_ids = {
        node_id for node_id, node in graph_nodes.items() if isinstance(node, Spread) or len(edges.get(node_id, [])) > 1
    }
    finder = DominatingForkFinder(
        nodes=nodes,
        start_ids=start_ids,
        fork_ids=fork_ids,
        edges=edges,
    )

    join_ids = {node.id for node in graph_nodes.values() if isinstance(node, Join)}
    dominating_forks: dict[JoinId, DominatingFork[NodeId]] = {}
    for join_id in join_ids:
        dominating_fork = finder.find_dominating_fork(join_id)
        if dominating_fork is None:
            # TODO: Print out the mermaid graph and explain the problem
            raise ValueError(f'Join node {join_id} has no dominating fork')
        dominating_forks[join_id] = dominating_fork

    return dominating_forks


@dataclass
class Graph[StateT, InputT, OutputT]:
    # deps_type: type[Any]
    state_type: type[StateT]
    input_type: type[InputT]
    output_type: type[OutputT]

    nodes: dict[NodeId, AnyNode]
    edges_by_source: dict[NodeId, list[Edge]]
    dominating_forks: dict[JoinId, DominatingFork[NodeId]]  # mapping from join node to the dominating fork

    @property
    def start_edges(self) -> list[Edge]:
        return self.edges_by_source.get(START.id, [])

    def get_dominating_fork(self, join_id: JoinId) -> DominatingFork[NodeId]:
        result = self.dominating_forks.get(join_id)
        if result is None:
            raise RuntimeError(f'Node {join_id} is not a join node or did not have a dominating fork (this is a bug)')
        return result


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


@dataclass
class Some[T]:
    value: T


type Maybe[T] = Some[T] | None  # like optional, but you can tell the difference between "no value" and "value is None"


@dataclass
class GraphRun[StateT, InputT, OutputT]:
    graph: Graph[StateT, InputT, OutputT]
    state: StateT
    inputs: InputT

    result: Maybe[OutputT] = None
    active_walks: list[GraphWalkState] = field(init=False)

    active_reducers: dict[tuple[NodeId, NodeRunId], Reducer[StateT, Any, Any]] = field(init=False)
    """The node id is for the join, the node run id is for the dominating fork."""
    # persistence: Any  # TODO: Implement use of this

    def __post_init__(self):
        self.active_walks = [
            GraphWalkState(node_id=START.id, context_inputs=self.inputs, node_inputs=self.inputs, fork_stack=())
        ]
        self.active_reducers = {}

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
                self._handle_spread(node, walk)
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

        return self.result

    def _handle_start(self, walk: GraphWalkState) -> None:
        # nothing to do besides start the graph
        self._handle_edges(walk, walk.context_inputs, walk.node_inputs)

    def _handle_step(self, step: Step[Any, Any, Any], walk: GraphWalkState):
        step_context = StepContext(self.state, walk.context_inputs)
        output = step.call(step_context)
        self._handle_edges(walk, output, output)

    def _handle_reduce_join(self, join: Join[Any, Any, Any], walk: GraphWalkState) -> None:
        # Find the matching fork run id in the stack; this will be used to look for an active reducer
        parent_fork = self.graph.get_dominating_fork(join.id)
        matching_fork_run_id = next(iter(x[1] for x in walk.fork_stack[::-1] if x[0] == parent_fork.fork_id), None)
        if matching_fork_run_id is None:
            raise RuntimeError(
                f'Fork {parent_fork.fork_id} not found in stack {walk.fork_stack}. This means the dominating fork is not dominating (this is a bug).'
            )

        # Get or create the active reducer
        reducer = self.active_reducers.get((join.id, matching_fork_run_id))
        if reducer is None:
            reducer = join.reducer_factory(self.state, walk.node_inputs)
            self.active_reducers[(join.id, matching_fork_run_id)] = reducer

        # Reduce
        reducer.reduce(self.state, walk.node_inputs)

    def _handle_spread(self, spread: Spread[Any, Any, Any], walk: GraphWalkState):
        self._handle_edges(walk, walk.context_inputs, walk.node_inputs)

    def _handle_decision(self, decision: Decision[Any, Any], walk: GraphWalkState) -> None:
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
                    ctx = TransformContext(self.state, walk.context_inputs, node_inputs)
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
                    output = reducer.finalize(self.state)
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
                    transform_context = TransformContext(self.state, context_inputs, next_node_inputs)
                    next_node_inputs = edge.transform(transform_context)

                self.active_walks.append(
                    GraphWalkState(edge.destination_id, context_inputs, next_node_inputs, fork_stack)
                )
        elif isinstance(node, Spread):
            for edge in edges:
                for item in next_node_inputs:
                    if edge.transform is not None:
                        transform_context = TransformContext(self.state, context_inputs, item)
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
