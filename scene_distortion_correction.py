import time

import cv2
import numpy as np

import repo.distortion
from app_tk.app import Application
from app_tk.component.button import ButtonComponent
from app_tk.component.component import Component
from app_tk.component.label import LabelComponent
from app_tk.component.spacer import SpacerComponent
from app_tk.component.spin_box import SpinBoxComponent
from app_tk.event import KeyEvent
from app_tk.key import Key
from model import DistortionParameters, DistortionCorrectionProfile
from scene_base import MyScene
from scene_save_profile import SaveProfileDelegate, SaveProfileScene


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
    def __init__(self, app: "Application"):
        super().__init__(app)

        self._points = DistortionCorrectionPoints(cooling_down_time_seconds=1, max_snapshots=20)
        self._is_detection_enabled = False

    def load_event(self):
        self.add_component(LabelComponent, "Distortion Correction", bold=True)
        self.add_component(
            LabelComponent,
            "Show the checkerboard to the camera and move it slightly across the entire scene\n"
            "Press <SPACE> to start",
        )
        self.add_component(SpacerComponent)
        self.add_component(LabelComponent, "", name="info")
        self.add_component(SpacerComponent)
        self.add_component(LabelComponent, "Number of intersections X")
        self.add_component(
            SpinBoxComponent,
            min_value=3,
            max_value=20,
            value=7,
            name="num-intersections-1",
        )
        self.add_component(LabelComponent, "Number of intersections Y")
        self.add_component(
            SpinBoxComponent,
            min_value=3,
            max_value=20,
            value=8,
            name="num-intersections-2",
        )
        self.add_component(ButtonComponent, "Start", name="b-toggle-detection")
        self.add_component(ButtonComponent, "Calculate Parameters and Save Profile",
                           name="b-save-profile")

    def update(self):
        self.find_component(LabelComponent, "info").set_text(
            f"Snapshot count: "
            f"{self._points.get_snapshot_count():2d}/{self._points.get_max_snapshot_count():2d}"
        )

    CORNER_SUB_PIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def render_canvas(self) -> np.ndarray | None:
        canvas = super().render_canvas()
        if canvas is None:
            return None

        if self._is_detection_enabled and not self._points.is_cooling_down():
            gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
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
                    self.get_app().make_toast(
                        "info",
                        "You got enough snapshots"
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

    def on_button_triggered(self, sender: Component) -> None:
        if isinstance(sender, ButtonComponent):
            if sender.get_name() == "b-toggle-detection":
                self.toggle_detection_enabled()
                return
            if sender.get_name() == "b-save-profile":
                self.get_app().make_toast(
                    "info",
                    "Calculating distortion parameters... please wait.",
                )
                width = self.get_app().camera_info.actual_spec.width
                height = self.get_app().camera_info.actual_spec.height
                try:
                    params = self._points.calculate_parameters(width, height)
                except ValueError as e:
                    self.get_app().make_toast(
                        "error",
                        e.args[0],
                    )
                else:
                    self.get_app().make_toast(
                        "info",
                        "Calculating distortion parameters... DONE",
                    )
                    delegator = DistortionCorrectionSaveProfileDelegate(params)
                    self.get_app().move_to(SaveProfileScene(self.get_app(), delegator))
                return
