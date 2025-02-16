from typing import cast

import cv2
import numpy as np

import repo.global_config
from core.tk.app import ApplicationWindowSize
from core.tk.component.button import ButtonComponent
from core.tk.component.component import Component
from core.tk.component.label import LabelComponent
from core.tk.component.separator import SeparatorComponent
from core.tk.component.spin_box import SpinBoxComponent
from core.tk.event import KeyEvent
from core.tk.global_state import get_app
from model.camera_param import CameraParamProfile
from model.distortion import DistortionProfile
from model.global_config import GlobalConfig, ROI
from model.laser_detection import LaserDetectionProfile
from model.laser_param import LaserParamProfile
from my_app import MyApplication
from scene.my_scene import MyScene
from util.light_section_method import get_laser_2d_and_3d_points


class MeasScene(MyScene):
    def __init__(
            self,
            distortion_profile: DistortionProfile,
            camera_param_profile: CameraParamProfile,
            laser_param_profile: LaserParamProfile,
            laser_detection_profile: LaserDetectionProfile,
    ):
        super().__init__()

        self._distortion_profile = distortion_profile  # TODO: unused, see method _get_last_image()
        self._camera_param_profile = camera_param_profile
        self._laser_param_profile = laser_param_profile
        self._laser_detection_profile = laser_detection_profile

    @classmethod
    def _get_last_image(cls) -> np.ndarray:
        app: MyApplication = cast(MyApplication, get_app())
        return app.last_capture_undistort.frame

    def _get_image_and_mask(self) -> tuple[np.ndarray, np.ndarray]:
        im = self._get_last_image()
        mask = self._laser_detection_profile.model.create_laser_mask(im, is_hsv=False)
        return im, mask

    def load_event(self):
        self.add_component(LabelComponent(self, "Meas", bold=True))
        self.add_component(SeparatorComponent(self))

        roi: ROI = repo.global_config.get().roi
        self.add_component(LabelComponent(self, "ROI Screen X Min"))
        self.add_component(
            SpinBoxComponent(
                self,
                value=roi.screen_x_min,
                min_value=0,
                max_value=2000,
                step=20,
                name="sp-roi-screen-x-min",
            )
        )
        self.add_component(LabelComponent(self, "ROI Screen X Max"))
        self.add_component(
            SpinBoxComponent(
                self,
                value=roi.screen_x_max,
                min_value=0,
                max_value=2000,
                step=20,
                name="sp-roi-screen-x-max",
            )
        )
        self.add_component(LabelComponent(self, "ROI Screen Y Min"))
        self.add_component(
            SpinBoxComponent(
                self,
                value=roi.screen_y_min,
                min_value=0,
                max_value=2000,
                step=20,
                name="sp-roi-screen-y-min",
            )
        )
        self.add_component(LabelComponent(self, "ROI Screen Y Max"))
        self.add_component(
            SpinBoxComponent(
                self,
                value=roi.screen_y_max,
                min_value=0,
                max_value=2000,
                step=20,
                name="sp-roi-screen-y-max",
            )
        )
        self.add_component(LabelComponent(self, "ROI World X Min"))
        self.add_component(
            SpinBoxComponent(
                self,
                value=roi.world_x_min,
                min_value=-1000,
                max_value=1000,
                step=5,
                name="sp-roi-world-x-min",
            )
        )
        self.add_component(LabelComponent(self, "ROI World X Max"))
        self.add_component(
            SpinBoxComponent(
                self,
                value=roi.world_x_max,
                min_value=-1000,
                max_value=1000,
                step=5,
                name="sp-roi-world-x-max",
            )
        )
        self.add_component(LabelComponent(self, "ROI World Y Min"))
        self.add_component(
            SpinBoxComponent(
                self,
                value=roi.world_y_min,
                min_value=-1000,
                max_value=1000,
                step=5,
                name="sp-roi-world-y-min",
            )
        )
        self.add_component(LabelComponent(self, "ROI World Y Max"))
        self.add_component(
            SpinBoxComponent(
                self,
                value=roi.world_y_max,
                min_value=-1000,
                max_value=1000,
                step=5,
                name="sp-roi-world-y-max",
            )
        )
        self.add_component(LabelComponent(self, "ROI World Z Min"))
        self.add_component(
            SpinBoxComponent(
                self,
                value=roi.world_z_min,
                min_value=-1000,
                max_value=1000,
                step=5,
                name="sp-roi-world-z-min",
            )
        )
        self.add_component(LabelComponent(self, "ROI World Z Max"))
        self.add_component(
            SpinBoxComponent(
                self,
                value=roi.world_z_max,
                min_value=-1000,
                max_value=1000,
                step=5,
                name="sp-roi-world-z-max",
            )
        )
        self.add_component(SeparatorComponent(self))
        self.add_component(ButtonComponent(self, "Back", name="b-back"))

    def unload_event(self):
        roi: ROI = self._get_roi()
        global_config: GlobalConfig = repo.global_config.get()
        global_config.roi = roi
        repo.global_config.put(global_config)

    def _get_roi(self) -> ROI:
        roi_screen_x_min = self.find_component(SpinBoxComponent, "sp-roi-screen-x-min").get_value()
        roi_screen_x_max = self.find_component(SpinBoxComponent, "sp-roi-screen-x-max").get_value()
        roi_screen_y_min = self.find_component(SpinBoxComponent, "sp-roi-screen-y-min").get_value()
        roi_screen_y_max = self.find_component(SpinBoxComponent, "sp-roi-screen-y-max").get_value()
        roi_world_x_min = self.find_component(SpinBoxComponent, "sp-roi-world-x-min").get_value()
        roi_world_x_max = self.find_component(SpinBoxComponent, "sp-roi-world-x-max").get_value()
        roi_world_y_min = self.find_component(SpinBoxComponent, "sp-roi-world-y-min").get_value()
        roi_world_y_max = self.find_component(SpinBoxComponent, "sp-roi-world-y-max").get_value()
        roi_world_z_min = self.find_component(SpinBoxComponent, "sp-roi-world-z-min").get_value()
        roi_world_z_max = self.find_component(SpinBoxComponent, "sp-roi-world-z-max").get_value()
        return ROI(
            screen_x_min=roi_screen_x_min,
            screen_x_max=roi_screen_x_max,
            screen_y_min=roi_screen_y_min,
            screen_y_max=roi_screen_y_max,
            world_x_min=roi_world_x_min,
            world_x_max=roi_world_x_max,
            world_y_min=roi_world_y_min,
            world_y_max=roi_world_y_max,
            world_z_min=roi_world_z_min,
            world_z_max=roi_world_z_max,
        )

    def create_background(self, window_size: ApplicationWindowSize) -> np.ndarray | None:
        im, mask = self._get_image_and_mask()

        # ROI
        roi: ROI = self._get_roi()
        im[:roi.screen_y_max, :roi.screen_x_min] //= 2
        im[:roi.screen_y_min, roi.screen_x_min:] //= 2
        im[roi.screen_y_min:, roi.screen_x_max:] //= 2
        im[roi.screen_y_max:, :roi.screen_x_max] //= 2

        # Solve screen and world points
        points_screen, points_world = get_laser_2d_and_3d_points(
            mask,
            self._camera_param_profile.param,
            self._laser_param_profile.param,
            roi,
        )
        if points_screen.size != 0:
            v_show = np.arange(10, im.shape[0] - 10, 30)
            vs_src = points_screen[:, 1]
            v_mask = np.in1d(vs_src, v_show)
            us, vs = points_screen[v_mask, :].T
            xs, ys, zs = points_world[v_mask, :].T
            for u, v, x, y, z in zip(us, vs, xs, ys, zs):
                # 水平線
                cv2.line(
                    im,
                    (0, int(v)),
                    (im.shape[1], int(v)),
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

                # 目盛り
                cx, _ = self._camera_param_profile.param.conversion_factor(u, v, z)
                for x_mm in range(-200, 201, 10):
                    x_ticker = u + x_mm / cx
                    if x_mm % 100 == 0:
                        h = 15
                    elif x_mm % 50 == 0:
                        h = 10
                    else:
                        h = 3
                    cv2.line(
                        im,
                        (int(x_ticker), int(v - h)),
                        (int(x_ticker), int(v + h)),
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

                # レーザーの位置に垂直線
                cv2.line(
                    im,
                    (int(u), int(v - 10)),
                    (int(u), int(v + 10)),
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                # レーザーの3D座標
                cv2.putText(
                    im,
                    f"({x:6.1f}, {y:6.1f}, {z:6.1f})",
                    (u + 5, v - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

            # 断面形状 ピクチャインピクチャ
            im_real = np.zeros((600, 600, 3), np.uint8)
            for x in range(100, 600, 100):
                cv2.line(
                    im_real,
                    (x, 0),
                    (x, 600),
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            for v in range(100, 600, 100):
                cv2.line(
                    im_real,
                    (0, v),
                    (600, v),
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            for i in range(len(points_world) - 1):
                x1, y1, z1 = points_world[i, :]
                x2, y2, z2 = points_world[i + 1, :]
                if abs(z2 - z1) >= 10:
                    continue
                cv2.line(
                    im_real,
                    (300 + int(z1), 300 - int(y1)),
                    (300 + int(z2), 300 - int(y2)),
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            self.set_picture_in_picture(im_real, width=400)
        else:
            self.set_picture_in_picture(None)

        return window_size.coerce(im)

    def key_event(self, event: KeyEvent) -> bool:
        if event.down:
            pass
        return super().key_event(event)

    def _on_button_triggered(self, sender: Component) -> None:
        if sender.get_name() == "b-back":
            get_app().move_back()
            return
        super()._on_button_triggered(sender)
