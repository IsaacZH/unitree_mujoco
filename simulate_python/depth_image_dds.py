from dataclasses import dataclass
import time

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types
import numpy as np

from unitree_sdk2py.idl.builtin_interfaces.msg.dds_ import Time_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import Header_


@dataclass
@annotate.final
@annotate.autoid("sequential")
class DepthIntrinsics_(idl.IdlStruct, typename="DepthIntrinsics_"):
    fx: types.float64
    fy: types.float64
    cx: types.float64
    cy: types.float64


@dataclass
@annotate.final
@annotate.autoid("sequential")
class DepthImage_(idl.IdlStruct, typename="DepthImage_"):
    header: Header_
    width: types.uint32
    height: types.uint32
    depth_scale: types.float32
    intrinsics: DepthIntrinsics_
    data: types.sequence[types.uint8]


def create_depth_message(depth_image, depth_scale, intrinsics, frame_id="depth_camera"):
    depth_array = np.asarray(depth_image, dtype=np.uint16)
    if depth_array.ndim != 2:
        raise ValueError("depth_image must be a 2D array")

    fx, fy, cx, cy = intrinsics
    height, width = depth_array.shape
    t = time.time()

    return DepthImage_(
        header=Header_(
            stamp=Time_(sec=int(t), nanosec=int((t % 1) * 1e9)),
            frame_id=frame_id,
        ),
        width=width,
        height=height,
        depth_scale=float(depth_scale),
        intrinsics=DepthIntrinsics_(
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
        ),
        data=depth_array.tobytes(),
    )


def decode_depth_message(msg: DepthImage_):
    depth = np.frombuffer(bytes(msg.data), dtype=np.uint16)
    return depth.reshape((int(msg.height), int(msg.width)))


def decode_intrinsics(msg: DepthImage_):
    return (
        float(msg.intrinsics.fx),
        float(msg.intrinsics.fy),
        float(msg.intrinsics.cx),
        float(msg.intrinsics.cy),
    )
