from __future__ import annotations

from typing import Any, Protocol


class TransformContext[StateT, DepsT, InputT, OutputT]:
    """The main reason this is not a dataclass is that we need it to be covariant in its type parameters."""

    def __init__(self, state: StateT, deps: DepsT, inputs: InputT, output: OutputT):
        self._state = state
        self._deps = deps
        self._inputs = inputs
        self._output = output

    @property
    def state(self) -> StateT:
        return self._state

    @property
    def deps(self) -> DepsT:
        return self._deps

    @property
    def inputs(self) -> InputT:
        return self._inputs

    @property
    def output(self) -> OutputT:
        return self._output

    def __repr__(self):
        return f'{self.__class__.__name__}(state={self.state}, deps={self.deps}, inputs={self.inputs}, output={self.output})'


class TransformFunction[StateT, DepsT, SourceInputT, SourceOutputT, DestinationInputT](Protocol):
    def __call__(self, ctx: TransformContext[StateT, DepsT, SourceInputT, SourceOutputT]) -> DestinationInputT:
        raise NotImplementedError


type AnyTransformFunction = TransformFunction[Any, Any, Any, Any, Any]
