import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading

import cv2
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand

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
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def Close(self):
        if not self.running:
            return

        self.running = False
        cv2.destroyWindow(self.window_name)

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

        grayscale, info_text = self._normalize_depth(depth_image)
        colorized = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            colorized,
            info_text,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
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

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
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
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)

    depth_visualizer = None
    if config.ENABLE_DEPTH_VISUALIZATION:
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
                time.sleep(config.DEPTH_VISUALIZATION_DT)
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
