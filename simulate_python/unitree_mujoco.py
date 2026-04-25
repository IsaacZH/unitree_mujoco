import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading

import cv2
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand
from depth_image_dds import DepthImage_, create_depth_message

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
        viewer.sync()
        locker.release()
        time.sleep(config.VIEWER_DT)


if __name__ == "__main__":
    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)

    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)

    depth_visualizer = None
    try:
        depth_visualizer = DepthVisualizer(
            mj_model, mj_data, config.DEPTH_CAMERA_NAME
        )
    except Exception as exc:
        print(f"Depth visualization disabled: {exc}")

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
