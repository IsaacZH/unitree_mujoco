from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types
import numpy as np


@dataclass
@annotate.final
@annotate.autoid("sequential")
class NavTarget_(idl.IdlStruct, typename="NavTarget_"):
    stamp_sec: types.uint32
    stamp_nanosec: types.uint32
    target_world: types.sequence[types.float32]


def decode_nav_target_message(msg: NavTarget_):
    target_world = np.asarray(msg.target_world, dtype=np.float32).ravel()
    if target_world.shape[0] < 3:
        target_world = np.pad(target_world, (0, 3 - target_world.shape[0]))
    return target_world[:3]

