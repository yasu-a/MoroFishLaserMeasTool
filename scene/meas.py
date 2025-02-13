from functools import cache
from typing import cast

import cv2
import numpy as np

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
from model.laser_detection import LaserDetectionProfile
from model.laser_param import LaserParamProfile
from my_app import MyApplication
from scene.my_scene import MyScene


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

        self._laser_x_on_vertical_smooth = None

    @classmethod
    def _get_last_image(cls) -> np.ndarray:
        app: MyApplication = cast(MyApplication, get_app())
        return app.last_capture_undistort.frame

    def _get_image_and_mask(self) -> tuple[np.ndarray, np.ndarray]:
        im = self._get_last_image()
        x1, y1, x2, y2 = self._get_roi()
        im_roi = im[y1:y2, x1:x2]
        mask_roi = self._laser_detection_profile.model.create_laser_mask(im_roi, is_hsv=False)
        mask = np.zeros((im.shape[0], im.shape[1]), np.uint8)
        mask[y1:y2, x1:x2] = mask_roi
        return im, mask

    def create_background(self, window_size: ApplicationWindowSize) -> np.ndarray | None:
        im, mask = self._get_image_and_mask()

        # ROI
        x1, y1, x2, y2 = self._get_roi()
        im[:y2, :x1] //= 2
        im[:y1, x1:] //= 2
        im[y1:, x2:] //= 2
        im[y2:, :x2] //= 2

        xs = np.arange(im.shape[1])

        laser_x_on_vertical \
            = (mask * xs[None, :]).sum(axis=1) / mask.sum(axis=1)  # nan if undetected
        if self._laser_x_on_vertical_smooth is None:
            self._laser_x_on_vertical_smooth = laser_x_on_vertical
        else:
            na = np.isnan(self._laser_x_on_vertical_smooth)
            self._laser_x_on_vertical_smooth[na] = laser_x_on_vertical[na]
            alpha = np.clip(
                np.abs(self._laser_x_on_vertical_smooth[~na] - laser_x_on_vertical[~na]) / 20,
                0.2,
                0.4,
            )
            self._laser_x_on_vertical_smooth[~na] \
                = self._laser_x_on_vertical_smooth[~na] * (1 - alpha) \
                  + laser_x_on_vertical[~na] * alpha

        @cache
        def get_p3d(y: int) -> tuple[float, float, float]:
            x_laser = self._laser_x_on_vertical_smooth[y]
            return self._camera_param_profile.param.from_2d_to_3d(
                x_laser,
                y,
                *self._laser_param_profile.param.vec[:3],
            )

        for y in range(10, im.shape[0] - 10, 30):
            x_laser = self._laser_x_on_vertical_smooth[y]
            if np.isnan(x_laser):
                continue

            # 水平線
            cv2.line(
                im,
                (int(xs[0]), int(y)),
                (int(xs[-1]), int(y)),
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

            p3d = get_p3d(y)
            x_laser = int(x_laser)

            # 目盛り
            cx, _ = self._camera_param_profile.param.conversion_factor(p3d[2])
            for x_mm in range(-200, 201, 10):
                x_ticker = x_laser + x_mm / cx
                if x_mm % 100 == 0:
                    h = 15
                elif x_mm % 50 == 0:
                    h = 10
                else:
                    h = 3
                cv2.line(
                    im,
                    (int(x_ticker), int(y - h)),
                    (int(x_ticker), int(y + h)),
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

            # レーザーの位置に垂直線
            cv2.line(
                im,
                (int(x_laser), int(y - 10)),
                (int(x_laser), int(y + 10)),
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            # レーザーの3D座標
            cv2.putText(
                im,
                f"({p3d[0]:6.1f}, {p3d[1]:6.1f}, {p3d[2]:6.1f})",
                (x_laser + 5, y - 5),
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
        for y in range(100, 600, 100):
            cv2.line(
                im_real,
                (0, y),
                (600, y),
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        for y1 in range(10, im.shape[0] - 10, 1):
            y2 = y1 + 1
            x1 = self._laser_x_on_vertical_smooth[y1]
            x2 = self._laser_x_on_vertical_smooth[y2]
            if np.isnan(x1) or np.isnan(x2):
                continue
            p3d_1, p3d_2 = get_p3d(y1), get_p3d(y2)
            cv2.line(
                im_real,
                (300 + int(p3d_1[2]), 300 - int(p3d_1[1])),
                (300 + int(p3d_2[2]), 300 - int(p3d_2[1])),
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        self.set_picture_in_picture(im_real, width=400)

        return window_size.coerce(im)

    def load_event(self):
        self.add_component(LabelComponent(self, "Meas", bold=True))
        self.add_component(SeparatorComponent(self))
        im = self._get_last_image()
        self.add_component(LabelComponent(self, "ROI X Min"))
        self.add_component(
            SpinBoxComponent(
                self,
                value=0,
                min_value=0,
                max_value=im.shape[1],
                step=10,
                name="sp-roi-x-min",
            )
        )
        self.add_component(LabelComponent(self, "ROI X Max"))
        self.add_component(
            SpinBoxComponent(
                self,
                value=im.shape[1],
                min_value=0,
                max_value=im.shape[1],
                step=10,
                name="sp-roi-x-max",
            )
        )
        self.add_component(LabelComponent(self, "ROI Y Min"))
        self.add_component(
            SpinBoxComponent(
                self,
                value=0,
                min_value=0,
                max_value=im.shape[0],
                step=10,
                name="sp-roi-y-min",
            )
        )
        self.add_component(LabelComponent(self, "ROI Y Max"))
        self.add_component(
            SpinBoxComponent(
                self,
                value=im.shape[0],
                min_value=0,
                max_value=im.shape[0],
                step=10,
                name="sp-roi-y-max",
            )
        )
        self.add_component(SeparatorComponent(self))
        self.add_component(ButtonComponent(self, "Back", name="b-back"))

    def _get_roi(self) -> tuple[int, int, int, int]:  # x1, y1, x2, y2
        roi_x_min = self.find_component(SpinBoxComponent, "sp-roi-x-min").get_value()
        roi_x_max = self.find_component(SpinBoxComponent, "sp-roi-x-max").get_value()
        roi_y_min = self.find_component(SpinBoxComponent, "sp-roi-y-min").get_value()
        roi_y_max = self.find_component(SpinBoxComponent, "sp-roi-y-max").get_value()
        return roi_x_min, roi_y_min, roi_x_max, roi_y_max

    def key_event(self, event: KeyEvent) -> bool:
        if event.down:
            pass
        return super().key_event(event)

    def _on_button_triggered(self, sender: Component) -> None:
        if sender.get_name() == "b-back":
            get_app().move_back()
            return
        super()._on_button_triggered(sender)
