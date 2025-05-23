from __future__ import annotations

from typing import NewType

NodeId = NewType('NodeId', str)
NodeRunId = NewType('NodeRunId', str)

JoinId = NewType('JoinId', NodeId)
ForkId = NewType('ForkId', NodeId)
