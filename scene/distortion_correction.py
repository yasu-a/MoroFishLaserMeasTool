import time
from typing import cast

import cv2
import numpy as np

import repo.distortion
from camera_server import CaptureResult
from core.tk.app import ApplicationWindowSize
from core.tk.component.button import ButtonComponent
from core.tk.component.component import Component
from core.tk.component.global_state import get_app
from core.tk.component.label import LabelComponent
from core.tk.component.spacer import SpacerComponent
from core.tk.component.spin_box import SpinBoxComponent
from core.tk.component.toast import Toast
from core.tk.event import KeyEvent
from core.tk.key import Key
from model import DistortionParameters, DistortionCorrectionProfile
from my_app import MyApplication
from scene.my_scene import MyScene
from scene.save_profile import SaveProfileDelegate, SaveProfileScene


class DistortionCorrectionPoints:
    def __init__(self, cooling_down_time_seconds: float, max_snapshots: int):
        self._cooling_down_time_seconds = cooling_down_time_seconds
        self._max_snapshots = max_snapshots

        self._obj_points = []  # 3d point in real world space
        self._img_points = []  # 2d points in image plane.
        self._timestamps = []

    def add_snapshot(self, obj_p, corner_p):
        self._obj_points.append(obj_p)
        self._img_points.append(corner_p)
        self._timestamps.append(time.time())

    def get_max_snapshot_count(self) -> int:
        return self._max_snapshots

    def get_snapshot_count(self) -> int:
        return len(self._timestamps)

    def is_cooling_down(self) -> bool:
        if self.get_snapshot_count():
            return time.time() - self._timestamps[-1] < self._cooling_down_time_seconds
        else:
            return False

    def is_more_points_needed(self) -> bool:
        return self.get_snapshot_count() < self.get_max_snapshot_count()

    def calculate_parameters(self, width: int, height: int) -> DistortionParameters:
        if self.get_snapshot_count() <= 5:
            raise ValueError("You need more snapshots")
        # print(f"obj_points={self._obj_points}")
        # print(f"img_points={self._img_points}")
        # print(f"{width=}")
        # print(f"{height=}")
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


class DistortionCorrectionSaveProfileDelegate(SaveProfileDelegate):
    def __init__(self, params: DistortionParameters):
        self._params = params

    def check_exist(self, name: str) -> bool:
        return repo.distortion.exists(name)

    def execute(self, name: str) -> str | None:
        profile = DistortionCorrectionProfile(
            name=name,
            params=self._params,
        )
        return repo.distortion.put(profile)


class DistortionCorrectionScene(MyScene):
    def __init__(self):
        super().__init__()

        self._points = DistortionCorrectionPoints(cooling_down_time_seconds=1, max_snapshots=20)
        self._is_detection_enabled = False

    def load_event(self):
        self.add_component(LabelComponent(self, "Distortion Correction", bold=True))
        self.add_component(
            LabelComponent(
                self,
                "Show the checkerboard to the camera and move it slightly across the entire scene\n"
                "Press <SPACE> to start",
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
        self.add_component(ButtonComponent(self, "Start", name="b-toggle-detection"))
        self.add_component(
            ButtonComponent(
                self,
                "Calculate Parameters and Save Profile",
                name="b-save-profile",
            )
        )
        self.add_component(SpacerComponent(self))
        self.add_component(ButtonComponent(self, "Back", name="b-back"))

    def update(self):
        self.find_component(LabelComponent, "info").set_text(
            f"Snapshot count: "
            f"{self._points.get_snapshot_count():2d}/{self._points.get_max_snapshot_count():2d}"
        )

    CORNER_SUB_PIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def create_background(self, window_size: "ApplicationWindowSize") -> np.ndarray | None:
        canvas = super().create_background(window_size)
        if canvas is None:
            return None

        # チェッカーボードを検出して検出結果を描画する
        if self._is_detection_enabled and not self._points.is_cooling_down():
            last_capture: CaptureResult | None = cast(MyApplication, get_app()).last_capture
            gray = cv2.cvtColor(last_capture.frame, cv2.COLOR_BGR2GRAY)
            s1 = self.find_component(SpinBoxComponent, "num-intersections-1").get_value()
            s2 = self.find_component(SpinBoxComponent, "num-intersections-2").get_value()
            ret, corners = cv2.findChessboardCorners(gray, (s2, s1), None)
            if ret:
                objp = np.zeros((s1 * s2, 3), np.float32)
                objp[:, :2] = np.mgrid[0:s2, 0:s1].T.reshape(-1, 2)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                            self.CORNER_SUB_PIX_CRITERIA)
                self._points.add_snapshot(objp, corners2)
                if not self._points.is_more_points_needed():
                    get_app().make_toast(
                        Toast(
                            self,
                            "info",
                            "You got enough samples",
                        )
                    )
                    self.toggle_detection_enabled()
                canvas = cv2.drawChessboardCorners(canvas, (s2, s1), corners2, ret)

        # キャリブレーション実行中なら画面全体を枠で囲う
        if self._is_detection_enabled:
            if self._points.is_cooling_down():
                color = 255, 0, 0
            else:
                color = 0, 0, 255
            cv2.rectangle(
                canvas,
                (0, 0),
                (canvas.shape[1], canvas.shape[0]),
                color,
                5,
            )

        return canvas

    def toggle_detection_enabled(self):
        self._is_detection_enabled = not self._is_detection_enabled
        if self._is_detection_enabled:
            self.find_component(ButtonComponent, "b-toggle-detection").set_text("Stop detection")
        else:
            self.find_component(ButtonComponent, "b-toggle-detection").set_text("Start detection")

    def key_event(self, event: KeyEvent) -> bool:
        if super().key_event(event):
            return True
        if event.down:
            if event.key == Key.SPACE:
                self.toggle_detection_enabled()
                return True
        return False

    def _on_button_triggered(self, sender: Component) -> None:
        if isinstance(sender, ButtonComponent):
            if sender.get_name() == "b-toggle-detection":
                self.toggle_detection_enabled()
                return
            if sender.get_name() == "b-back":
                get_app().move_back()
                return
            if sender.get_name() == "b-save-profile":
                app = cast(MyApplication, get_app())
                app.make_toast(
                    Toast(
                        self,
                        "info",
                        "Calculating distortion parameters... please wait."
                    ),
                )
                if app.camera_info is None:
                    app.make_toast(
                        Toast(
                            self,
                            "error",
                            "No camera information available",
                        )
                    )
                    return
                width = app.camera_info.actual_spec.width
                height = app.camera_info.actual_spec.height
                try:
                    params = self._points.calculate_parameters(width, height)
                except ValueError as e:
                    app.make_toast(
                        Toast(
                            self,
                            "error",
                            e.args[0],
                        )
                    )
                else:
                    app.make_toast(
                        Toast(
                            self,
                            "info",
                            "Calculating distortion parameters... DONE",
                        )
                    )
                    delegator = DistortionCorrectionSaveProfileDelegate(params)
                    get_app().move_to(SaveProfileScene(delegator))
                return
