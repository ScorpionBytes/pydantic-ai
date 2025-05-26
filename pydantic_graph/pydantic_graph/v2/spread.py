from __future__ import annotations

from dataclasses import dataclass

from pydantic_graph.v2.id_types import ForkId


# TODO: remove these generic parameters? I think they aren't actually doing/used for anything
@dataclass
class Spread[StateT, DepsT, InputT, OutputT]:
    id: ForkId
