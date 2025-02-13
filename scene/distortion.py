import time
from typing import cast

import cv2
import numpy as np

import repo.distortion
from camera_server import CaptureResult
from core.tk.app import ApplicationWindowSize
from core.tk.component.button import ButtonComponent
from core.tk.component.check_box import CheckBoxComponent
from core.tk.component.component import Component
from core.tk.component.label import LabelComponent
from core.tk.component.spacer import SpacerComponent
from core.tk.component.spin_box import SpinBoxComponent
from core.tk.component.toast import Toast
from core.tk.dialog import InputNameDialog, MessageDialog
from core.tk.event import KeyEvent
from core.tk.global_state import get_app
from core.tk.key import Key
from core.tk.rendering import UIRenderingContext
from model.distortion import DistortionParameters, DistortionProfile
from my_app import MyApplication
from scene.my_scene import MyScene


class DistortionCorrectionPoints:
    def __init__(self, cooling_down_time_seconds: float, min_samples: int, max_samples: int):
        self._cooling_down_time_seconds = cooling_down_time_seconds
        self._min_samples = min_samples
        self._max_samples = max_samples

        self._obj_points = []  # 3d point in real world space
        self._img_points = []  # 2d points in image plane.
        self._timestamps = []

    def add_sample(self, obj_p, corner_p):
        self._obj_points.append(obj_p)
        self._img_points.append(corner_p)
        self._timestamps.append(time.time())

    def get_max_sample_count(self) -> int:
        return self._max_samples

    def get_sample_count(self) -> int:
        return len(self._timestamps)

    def is_cooling_down(self) -> bool:
        if self.get_sample_count():
            return time.time() - self._timestamps[-1] < self._cooling_down_time_seconds
        else:
            return False

    def is_more_points_needed(self) -> bool:
        return self.get_sample_count() < self.get_max_sample_count()

    def clear(self) -> None:
        self._obj_points.clear()
        self._img_points.clear()
        self._timestamps.clear()

    def calculate_parameters(self, width: int, height: int) -> DistortionParameters:
        if self.get_sample_count() < self._min_samples:
            raise ValueError(f"You need more than {self._min_samples} samples")
        (
            ret,
            mtx,
            dist,
            rvecs,
            tvecs
        ) = cv2.calibrateCamera(
            self._obj_points,
            self._img_points,
            (width, height),
            None,
            None,
        )
        # noinspection PyTypeChecker
        return DistortionParameters(
            ret=ret,
            mtx=mtx,
            dist=dist,
            rvecs=rvecs,
            tvecs=tvecs,
        )


class DistortionCorrectionScene(MyScene):
    def __init__(self):
        super().__init__()

        self._points = DistortionCorrectionPoints(
            cooling_down_time_seconds=1,
            min_samples=3,
            max_samples=30,
        )
        self._is_detection_enabled = False

        self._last_param: DistortionParameters | None = None
        self._last_time_preview = 0

    def load_event(self):
        self.add_component(LabelComponent(self, "Distortion Correction", bold=True))
        self.add_component(
            LabelComponent(
                self,
                "After press <SPACE> to start detection, show the checkerboard to the camera "
                "and move it slightly across the entire screen."
            ),
        )
        self.add_component(SpacerComponent(self))
        self.add_component(LabelComponent(self, "", name="info"))
        self.add_component(SpacerComponent(self))
        self.add_component(LabelComponent(self, "Number of intersections X"))
        self.add_component(
            SpinBoxComponent(
                self,
                7,
                min_value=3,
                max_value=20,
                name="num-intersections-1",
            )
        )
        self.add_component(LabelComponent(self, "Number of intersections Y"))
        self.add_component(
            SpinBoxComponent(
                self,
                8,
                min_value=3,
                max_value=20,
                name="num-intersections-2",
            )
        )
        self.add_component(SpacerComponent(self))
        self.add_component(
            ButtonComponent(self, "Start Detection <SPACE>", name="b-toggle-detection")
        )
        self.add_component(SpacerComponent(self))
        self.add_component(
            CheckBoxComponent(
                self,
                "Preview correction result if parameter exists",
                True,
                name="cb-preview",
            )
        )
        self.add_component(
            ButtonComponent(
                self,
                "Calculate Parameters and Save",
                name="b-calc-param-and-save",
            )
        )
        self.add_component(
            ButtonComponent(
                self,
                "Discard Samples",
                name="b-discard",
            )
        )
        self.add_component(SpacerComponent(self))
        self.add_component(ButtonComponent(self, "Back", name="b-back"))

    @classmethod
    def _get_last_capture(cls) -> CaptureResult | None:
        last_capture: CaptureResult | None = cast(MyApplication, get_app()).last_capture
        return last_capture

    def is_preview_enabled(self) -> bool:
        return self.find_component(CheckBoxComponent, "cb-preview").get_value()

    def update(self):
        # サンプルカウントを更新
        self.find_component(LabelComponent, "info").set_text(
            f"Sample count: "
            f"{self._points.get_sample_count():2d}/{self._points.get_max_sample_count():2d}"
        )

        # 計算済みパラメータが存在すれば一定間隔でpicture-in-pictureで表示
        now = time.monotonic()
        if now - self._last_time_preview >= 1:
            self._last_time_preview = now
            last_capture: CaptureResult | None = self._get_last_capture()
            if self._last_param is not None \
                    and last_capture is not None \
                    and self.is_preview_enabled():
                im_undistort = self._last_param.undistort(last_capture.frame)
                self.set_picture_in_picture(im_undistort, height=300)
            else:
                self.set_picture_in_picture(None)

    CORNER_SUB_PIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def create_background(self, window_size: "ApplicationWindowSize") -> np.ndarray | None:
        canvas = super().create_background(window_size)
        if canvas is None:
            return None

        # チェッカーボードを検出して検出結果を描画する
        if self._is_detection_enabled and not self._points.is_cooling_down():
            last_capture: CaptureResult | None = self._get_last_capture()
            gray = cv2.cvtColor(last_capture.frame, cv2.COLOR_BGR2GRAY)
            s1 = self.find_component(SpinBoxComponent, "num-intersections-1").get_value()
            s2 = self.find_component(SpinBoxComponent, "num-intersections-2").get_value()
            ret, corners = cv2.findChessboardCorners(gray, (s2, s1), None)
            if ret:
                objp = np.zeros((s1 * s2, 3), np.float32)
                objp[:, :2] = np.mgrid[0:s2, 0:s1].T.reshape(-1, 2)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                            self.CORNER_SUB_PIX_CRITERIA)
                self._points.add_sample(objp, corners2)
                if not self._points.is_more_points_needed():
                    get_app().make_toast(
                        Toast(
                            self,
                            "info",
                            "You've got enough samples!",
                        )
                    )
                    self.toggle_detection_enabled()
                canvas = cv2.drawChessboardCorners(canvas, (s2, s1), corners2, ret)

        return canvas

    def render_ui(self, ctx: UIRenderingContext) -> UIRenderingContext:
        # キャリブレーション実行中なら画面全体を枠で囲う
        if self._is_detection_enabled:
            if self._points.is_cooling_down():
                ctx.canvas.fullscreen_rect(ctx.style.border_normal)
            else:
                ctx.canvas.fullscreen_rect(ctx.style.border_abnormal)

        return super().render_ui(ctx)

    def toggle_detection_enabled(self):
        self._is_detection_enabled = not self._is_detection_enabled
        if self._is_detection_enabled:
            self.find_component(ButtonComponent, "b-toggle-detection").set_text(
                "Stop Detection <SPACE>"
            )
        else:
            self.find_component(ButtonComponent, "b-toggle-detection").set_text(
                "Start Detection <SPACE>"
            )

    def discard_samples(self):
        self._last_param = None
        self._points.clear()
        self.find_component(LabelComponent, "info").set_text(
            f"Snapshot count: 0/{self._points.get_max_sample_count():2d}"
        )

    def key_event(self, event: KeyEvent) -> bool:
        if event.down:
            if event.key == Key.SPACE:
                self.toggle_detection_enabled()
                return True
        if super().key_event(event):
            return True
        return False

    def _on_button_triggered(self, sender: Component) -> None:
        if isinstance(sender, ButtonComponent):
            if sender.get_name() == "b-toggle-detection":
                self.toggle_detection_enabled()
                if self._is_detection_enabled:
                    get_app().make_toast(
                        Toast(
                            self,
                            "info",
                            "Show the checkerboard to the camera!"
                        )
                    )
                return

            if sender.get_name() == "b-back":
                if self._points.get_sample_count() == 0:
                    get_app().move_back()
                    return
                else:
                    def callback(button_name: str | None) -> None:
                        get_app().close_dialog()
                        if button_name == "Yes":
                            get_app().move_back()

                    get_app().show_dialog(
                        MessageDialog(
                            is_error=True,
                            message=f"{self._points.get_sample_count()} samples are left. "
                                    f"Are you sure to exit?",
                            buttons=("No", "Yes"),
                            callback=callback,
                        )
                    )
                    return

            if sender.get_name() == "b-calc-param-and-save":
                app = cast(MyApplication, get_app())
                if app.camera_info is None:
                    app.make_toast(
                        Toast(
                            self,
                            "error",
                            "Failed to get parameters: no camera information available",
                        )
                    )
                    return
                width = app.camera_info.actual_spec.width
                height = app.camera_info.actual_spec.height
                try:
                    param: DistortionParameters = self._points.calculate_parameters(width, height)
                except ValueError as e:
                    self._last_param = None
                    get_app().make_toast(
                        Toast(
                            self,
                            "error",
                            e.args[0],
                        )
                    )
                else:
                    self._last_param = param

                    get_app().make_toast(
                        Toast(
                            self,
                            "info",
                            "Calculating distortion parameters completed",
                        )
                    )

                    def validator(name: str) -> str | None:
                        if name == "":
                            return "Please enter a name for the profile"
                        return None

                    def already_exist_checker(name: str) -> bool:
                        return repo.distortion.exists(name)

                    def callback(name: str | None) -> None:
                        get_app().close_dialog()

                        if name is None:
                            get_app().make_toast(
                                Toast(
                                    self,
                                    "error",
                                    "Canceled",
                                )
                            )
                            return
                        else:
                            profile = DistortionProfile(
                                name=name,
                                params=param,
                            )
                            repo.distortion.put(profile)
                            get_app().make_toast(
                                Toast(
                                    self,
                                    "info",
                                    f"Profile '{name}' saved successfully",
                                )
                            )

                    get_app().show_dialog(
                        InputNameDialog(
                            title="Save Distortion Profile",
                            validator=validator,
                            already_exist_checker=already_exist_checker,
                            callback=callback,
                        )
                    )

            if sender.get_name() == "b-discard":
                if self._points.get_sample_count() == 0:
                    get_app().make_toast(
                        Toast(
                            self,
                            "info",
                            "No samples are taken",
                        )
                    )
                    return
                else:
                    def callback(button_name: str | None) -> None:
                        get_app().close_dialog()
                        if button_name == "Yes":
                            self.discard_samples()
                            get_app().make_toast(
                                Toast(
                                    self,
                                    "info",
                                    "Samples are discarded",
                                )
                            )
                            return

                    get_app().show_dialog(
                        MessageDialog(
                            is_error=True,
                            message="Are you sure you want to discard samples?",
                            buttons=("No", "Yes"),
                            callback=callback,
                        )
                    )
                    return
