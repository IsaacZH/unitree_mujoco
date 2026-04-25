import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading

import cv2
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand
from depth_image_dds import DepthImage_, create_depth_message
from nav_debug_dds import NavDebug_, decode_nav_debug_message
from nav_target_dds import NavTarget_, decode_nav_target_message

import config


locker = threading.Lock()

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)


if config.ENABLE_ELASTIC_BAND:
    elastic_band = ElasticBand()
    if config.ROBOT == "h1" or config.ROBOT == "g1":
        band_attached_link = mj_model.body("torso_link").id
    else:
        band_attached_link = mj_model.body("base_link").id
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback
    )
else:
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = True
viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = True

mj_model.opt.timestep = config.SIMULATE_DT
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)


def downsample_and_crop_depth(depth_image, intrinsics, factor, target_width, target_height):
    fx, fy, cx, cy = intrinsics

    ds = depth_image[::factor, ::factor]
    fx_ds = fx / factor
    fy_ds = fy / factor
    cx_ds = cx / factor
    cy_ds = cy / factor

    h_ds, w_ds = ds.shape
    if target_width > w_ds or target_height > h_ds:
        raise ValueError(
            f"Target size {target_width}x{target_height} is larger than downsampled image {w_ds}x{h_ds}"
        )

    x0 = (w_ds - target_width) // 2
    y0 = (h_ds - target_height) // 2
    x1 = x0 + target_width
    y1 = y0 + target_height

    cropped = ds[y0:y1, x0:x1]
    cx_out = cx_ds - x0
    cy_out = cy_ds - y0
    return cropped, (fx_ds, fy_ds, cx_out, cy_out)


class DepthVisualizer:
    def __init__(self, mj_model, mj_data, camera_name):
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.camera_name = camera_name
        self.camera = mj_model.camera(camera_name)
        self.width, self.height = [
            int(value) for value in mj_model.cam_resolution[self.camera.id]
        ]
        if self.width <= 1 or self.height <= 1:
            raise ValueError(
                f"Camera '{camera_name}' must define a positive resolution for depth visualization"
            )

        self.renderer = mujoco.Renderer(mj_model, height=self.height, width=self.width)
        self.running = True
        self.window_name = f"Depth View: {camera_name}"
        self.far_clip = float(mj_model.stat.extent * mj_model.vis.map.zfar)
        fx, fy, principal_x_offset, principal_y_offset = [
            float(value) for value in mj_model.cam_intrinsic[self.camera.id]
        ]
        # MuJoCo principalpixel is stored as an offset from image center.
        cx = 0.5 * self.width + principal_x_offset
        cy = 0.5 * self.height + principal_y_offset
        self.camera_intrinsics = (fx, fy, cx, cy)
        self.last_publish_time = 0.0

        self.depth_publisher = ChannelPublisher("rt/depth_image", DepthImage_)
        self.depth_publisher.Init()

        if config.ENABLE_DEPTH_VISUALIZATION:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def Close(self):
        if not self.running:
            return

        self.running = False
        if config.ENABLE_DEPTH_VISUALIZATION:
            cv2.destroyWindow(self.window_name)
        if self.depth_publisher is not None:
            self.depth_publisher.Close()

    def _process_depth(self, depth_image):
        cropped_depth, intrinsics_out = downsample_and_crop_depth(
            depth_image,
            self.camera_intrinsics,
            factor=config.DEPTH_DDS_DOWNSAMPLE_FACTOR,
            target_width=config.DEPTH_DDS_WIDTH,
            target_height=config.DEPTH_DDS_HEIGHT,
        )

        processed_depth = np.asarray(cropped_depth, dtype=np.float32).copy()
        invalid = (~np.isfinite(processed_depth)) | (processed_depth <= 0) | (
            processed_depth >= self.far_clip - 1e-3
        )
        processed_depth[invalid] = 0.0

        depth_units = np.round(processed_depth / config.DEPTH_DDS_SCALE)
        depth_units = np.clip(depth_units, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        return processed_depth, depth_units, intrinsics_out

    def _normalize_depth(self, depth_image):
        depth = np.asarray(depth_image)
        valid = np.isfinite(depth) & (depth > 0)
        foreground = valid & (depth < self.far_clip - 1e-3)

        grayscale = np.zeros(depth.shape, dtype=np.uint8)
        if not np.any(foreground):
            return grayscale, "depth range: no valid hits"

        if config.DEPTH_AUTO_RANGE:
            foreground_depth = depth[foreground]
            near = float(np.percentile(foreground_depth, 2.0))
            far = float(np.percentile(foreground_depth, 98.0))
        else:
            near, far = config.DEPTH_RANGE

        far = max(far, near + 1e-6)
        clipped = np.clip(depth, near, far)
        grayscale[foreground] = np.round(
            (1.0 - (clipped[foreground] - near) / (far - near)) * 255.0
        ).astype(np.uint8)

        return grayscale, f"depth range: {near:.2f}m - {far:.2f}m"

    def Update(self):
        if not self.running:
            return False

        with locker:
            self.renderer.enable_depth_rendering()
            self.renderer.update_scene(self.mj_data, camera=self.camera_name)
            depth_image = self.renderer.render()
            self.renderer.disable_depth_rendering()

        processed_depth, depth_units, intrinsics_out = self._process_depth(depth_image)

        now = time.perf_counter()
        if now - self.last_publish_time >= config.DEPTH_DDS_DT:
            msg = create_depth_message(
                depth_units,
                depth_scale=config.DEPTH_DDS_SCALE,
                intrinsics=intrinsics_out,
                frame_id="depth_camera",
            )
            self.depth_publisher.Write(msg)
            self.last_publish_time = now

        if config.ENABLE_DEPTH_VISUALIZATION:
            grayscale, info_text = self._normalize_depth(processed_depth)
            colorized = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
            cv2.imshow(self.window_name, colorized)

            key = cv2.waitKey(1)
            if key == 27:
                self.Close()
                return False

            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                self.Close()

        return self.running


class NavDebugVisualizer:
    def __init__(self, mj_model, mj_data):
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.enabled = bool(config.ENABLE_NAV_DEBUG_VISUALIZATION)
        self._lock = threading.Lock()
        self._target_dir_b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self._target_speed_b = np.zeros(3, dtype=np.float32)
        self._target_world = np.zeros(3, dtype=np.float32)
        self._has_msg = False
        self._has_target = False

        self.base_body_id = mj_model.body(config.NAV_DEBUG_BASE_BODY).id
        self._sub = None
        self._target_sub = None
        if self.enabled:
            self._sub = ChannelSubscriber(config.NAV_DEBUG_TOPIC, NavDebug_)
            self._sub.Init(self._on_message, 10)
            self._target_sub = ChannelSubscriber(config.NAV_TARGET_TOPIC, NavTarget_)
            self._target_sub.Init(self._on_target_message, 10)
            print(f"Nav debug visualization subscribed: {config.NAV_DEBUG_TOPIC}")
            print(f"Nav target visualization subscribed: {config.NAV_TARGET_TOPIC}")

    def _on_message(self, msg: NavDebug_):
        target_dir_b, target_speed_b = decode_nav_debug_message(msg)
        with self._lock:
            self._target_dir_b = target_dir_b.astype(np.float32)
            self._target_speed_b = target_speed_b.astype(np.float32)
            self._has_msg = True

    def _on_target_message(self, msg: NavTarget_):
        target_world = decode_nav_target_message(msg)
        with self._lock:
            self._target_world = target_world.astype(np.float32)
            self._has_target = True

    def _snapshot(self):
        with self._lock:
            return (
                self._target_dir_b.copy(),
                self._target_speed_b.copy(),
                self._target_world.copy(),
                self._has_msg,
                self._has_target,
            )

    @staticmethod
    def _normalize(v):
        n = np.linalg.norm(v)
        if n < 1e-6:
            return None
        return v / n

    def _body_vec_to_world(self, vec_b):
        out = np.zeros(3, dtype=np.float64)
        quat = self.mj_data.xquat[self.base_body_id].astype(np.float64)
        mujoco.mju_rotVecQuat(out, vec_b.astype(np.float64), quat)
        return out.astype(np.float32)

    @staticmethod
    def _append_arrow(scene, start, end, radius, rgba):
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).flatten(),
            np.asarray(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            float(radius),
            np.asarray(start, dtype=np.float64),
            np.asarray(end, dtype=np.float64),
        )
        geom.rgba[:] = np.asarray(rgba, dtype=np.float32)
        scene.ngeom += 1

    @staticmethod
    def _append_box(scene, center, size, rgba):
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_BOX,
            np.asarray(size, dtype=np.float32),
            np.asarray(center, dtype=np.float32),
            np.eye(3).flatten(),
            np.asarray(rgba, dtype=np.float32),
        )
        geom.rgba[:] = np.asarray(rgba, dtype=np.float32)
        scene.ngeom += 1

    def UpdateScene(self, scene):
        if not self.enabled:
            scene.ngeom = 0
            return

        scene.ngeom = 0
        target_dir_b, target_speed_b, target_world, has_msg, has_target = self._snapshot()
        if not has_msg and not has_target:
            return

        origin = self.mj_data.xpos[self.base_body_id].copy()
        origin[2] += float(config.NAV_DEBUG_ARROW_Z_OFFSET)

        # Goal direction arrow (fixed length)
        dir_w = self._body_vec_to_world(target_dir_b)
        dir_w = self._normalize(dir_w)
        if dir_w is not None:
            end = origin + dir_w * float(config.NAV_DEBUG_TARGET_ARROW_LENGTH)
            self._append_arrow(
                scene,
                origin,
                end,
                config.NAV_DEBUG_TARGET_ARROW_RADIUS,
                config.NAV_DEBUG_TARGET_ARROW_RGBA,
            )

        # Speed command arrow (length scales with speed magnitude in xy plane)
        speed_b = target_speed_b.copy()
        speed_b[2] = 0.0
        speed_mag = float(np.linalg.norm(speed_b))
        speed_dir_w = self._normalize(self._body_vec_to_world(speed_b))
        if speed_dir_w is not None:
            speed_len = np.clip(
                speed_mag * float(config.NAV_DEBUG_SPEED_ARROW_SCALE),
                float(config.NAV_DEBUG_SPEED_ARROW_MIN),
                float(config.NAV_DEBUG_SPEED_ARROW_MAX),
            )
            end = origin + speed_dir_w * speed_len
            self._append_arrow(
                scene,
                origin,
                end,
                config.NAV_DEBUG_SPEED_ARROW_RADIUS,
                config.NAV_DEBUG_SPEED_ARROW_RGBA,
            )

        # Target world-position marker (yellow box)
        if has_target:
            marker_size = float(config.NAV_DEBUG_TARGET_BOX_SIZE)
            center = target_world.copy()
            center[2] += marker_size + float(config.NAV_DEBUG_TARGET_BOX_Z_LIFT)
            self._append_box(
                scene,
                center,
                np.array([marker_size, marker_size, marker_size], dtype=np.float32),
                config.NAV_DEBUG_TARGET_BOX_RGBA,
            )


def SimulationThread():
    global mj_data, mj_model
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)

    if config.USE_JOYSTICK:
        unitree.SetupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)
    if config.PRINT_SCENE_INFORMATION:
        unitree.PrintSceneInformation()

    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()

        # Update control torques from latest command
        unitree.UpdateControl()

        if config.ENABLE_ELASTIC_BAND:
            if elastic_band.enable:
                mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                    mj_data.qpos[:3], mj_data.qvel[:3]
                )
        mujoco.mj_step(mj_model, mj_data)

        locker.release()

        time_until_next_step = mj_model.opt.timestep - (
            time.perf_counter() - step_start
        )
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


def PhysicsViewerThread():
    while viewer.is_running():
        locker.acquire()
        if nav_debug_visualizer is not None:
            nav_debug_visualizer.UpdateScene(viewer.user_scn)
        viewer.sync()
        locker.release()
        time.sleep(config.VIEWER_DT)


if __name__ == "__main__":
    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)

    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)
    nav_debug_visualizer = None

    depth_visualizer = None
    try:
        depth_visualizer = DepthVisualizer(
            mj_model, mj_data, config.DEPTH_CAMERA_NAME
        )
    except Exception as exc:
        print(f"Depth visualization disabled: {exc}")

    try:
        nav_debug_visualizer = NavDebugVisualizer(mj_model, mj_data)
    except Exception as exc:
        print(f"Nav debug visualization disabled: {exc}")

    viewer_thread.start()
    sim_thread.start()

    try:
        while viewer.is_running():
            if depth_visualizer is not None and depth_visualizer.running:
                depth_visualizer.Update()
                time.sleep(min(config.DEPTH_VISUALIZATION_DT, config.DEPTH_DDS_DT))
            else:
                time.sleep(config.VIEWER_DT)
    except KeyboardInterrupt:
        viewer.close()
    finally:
        if depth_visualizer is not None:
            depth_visualizer.Close()
        cv2.destroyAllWindows()

        viewer_thread.join()
        sim_thread.join()
