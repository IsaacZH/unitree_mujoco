from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types
import numpy as np


@dataclass
@annotate.final
@annotate.autoid("sequential")
class NavDebug_(idl.IdlStruct, typename="NavDebug_"):
    stamp_sec: types.uint32
    stamp_nanosec: types.uint32
    target_dir_b: types.sequence[types.float32]
    target_speed_b: types.sequence[types.float32]


def decode_nav_debug_message(msg: NavDebug_):
    target_dir_b = np.asarray(msg.target_dir_b, dtype=np.float32).ravel()
    target_speed_b = np.asarray(msg.target_speed_b, dtype=np.float32).ravel()
    if target_dir_b.shape[0] < 3:
        target_dir_b = np.pad(target_dir_b, (0, 3 - target_dir_b.shape[0]))
    if target_speed_b.shape[0] < 3:
        target_speed_b = np.pad(target_speed_b, (0, 3 - target_speed_b.shape[0]))
    return target_dir_b[:3], target_speed_b[:3]
