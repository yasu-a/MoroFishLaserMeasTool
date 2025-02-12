from typing import Any, cast

import cv2
import numpy as np
import pandas as pd

import repo.laser_param
from app_logging import create_logger
from core.tk.color import Color
from core.tk.component.button import ButtonComponent
from core.tk.component.check_box import CheckBoxComponent
from core.tk.component.component import Component
from core.tk.component.label import LabelComponent
from core.tk.component.separator import SeparatorComponent
from core.tk.component.spacer import SpacerComponent
from core.tk.component.toast import Toast
from core.tk.dialog import InputNameDialog, MessageDialog
from core.tk.event import MouseEvent, KeyEvent
from core.tk.global_state import get_app
from core.tk.key import Key
from model.camera_param import CameraParamProfile, CameraParam
from model.image import Image
from model.laser_param import LaserParam, LaserParamProfile
from scene.laser_input import LaserInputScene, InputLine
from util.camera_calib_model import CameraCalibModel, DEFAULT_CALIB_MODEL
from util.space_alg import Plane3D


class LaserParamScene(LaserInputScene):
    _logger = create_logger()

    def __init__(self, *, image: Image, camera_param_profile: CameraParamProfile):
        super().__init__(image=image, n_sample_points_per_line=5)

        self._calib_model: CameraCalibModel = DEFAULT_CALIB_MODEL

        self._camera_param_profile: CameraParamProfile = camera_param_profile
        self._active_plane_index: int | None = None

    def load_event(self):
        self.add_component(LabelComponent(self, "Laser Parameter", bold=True))
        self.add_component(SeparatorComponent(self))
        self.add_component(
            CheckBoxComponent(
                self,
                "Snap to auto detected points (S)",
                True,
                name="cb-snap",
            )
        )
        self.add_component(
            CheckBoxComponent(
                self,
                "Show 3D model (W)",
                True,
                name="cb-show-3d-model",
            )
        )
        self.add_component(SeparatorComponent(self))
        self.add_component(LabelComponent(self, "", name="l-param"))
        self.add_component(ButtonComponent(self, "Save", name="b-save"))
        self.add_component(SpacerComponent(self))
        self.add_component(ButtonComponent(self, "Back", name="b-back"))

    def _is_snap_enabled(self) -> bool:
        return self.find_component(CheckBoxComponent, "cb-snap").get_value()

    def _is_show_3d_model_enabled(self) -> bool:
        return self.find_component(CheckBoxComponent, "cb-show-3d-model").get_value()

    def get_laser_param(self) -> LaserParam | None:
        if self._input_lines.get_point_count() <= 3:
            return None

        points = []
        for line in self._input_lines.iter_lines():
            plane_index: int = cast(int, line.user_data)
            plane = Plane3D.from_points(self._calib_model.get_plane_corners(plane_index))
            for u, v in line.points:
                x, y, z = self._camera_param_profile.param.from_2d_to_3d(
                    u, v, plane.a, plane.b, plane.c,
                )
                points.append([x, y, z])
        points = np.array(points)

        try:
            p, error = Plane3D.from_points(points, is_complete_plane=False, return_error=True)
            vec = np.array([p.a, p.b, p.c, 1.0])
            laser_param = LaserParam(vec=vec, error=error)
        except np.linalg.LinAlgError:
            return None
        else:
            self._logger.debug(
                f"Laser parameter solved\n"
                f"{pd.DataFrame(laser_param.vec).round(6)!s}\n"
                f"{pd.DataFrame(points).round(0)!s}"
            )
            return laser_param

    def update(self):
        if self._calib_model is not None and self._is_show_3d_model_enabled():
            plane_colors = {}
            if self._active_plane_index is not None:
                plane_colors[self._active_plane_index] = 0, 0, 255
            self.set_picture_in_picture(
                self._calib_model.render_3d(
                    width=500,
                    height=500,
                    plane_colors=plane_colors,
                ),
                500,
            )
        else:
            self.set_picture_in_picture(None)
        return super().update()

    def _get_line_color(self, line: InputLine) -> Color:
        plane_index: int = cast(int, line.user_data)
        if plane_index == self._active_plane_index:
            return Color.RED
        else:
            return Color.WHITE

    def _draw_on_background(self, im: np.ndarray) -> None:
        camera_pram: CameraParam = self._camera_param_profile.param
        if camera_pram is not None:
            def _proj(p):
                _p3d = np.array([*p, 1])
                _p2d = camera_pram.mat @ _p3d
                _p2d /= _p2d[2]
                return _p2d[:2].round(0).astype(int)

            # カメラパラメータ行列が張る空間の格子線の描画
            size = self._calib_model.get_size()
            for v in np.arange(0, size + 1e-6, 20):
                p1, p2 = np.array([v, 0, 0]), np.array([v, size, 0])
                for i in range(3):
                    cv2.line(im, _proj(p1), _proj(p2), (200, 50, 0), 1, cv2.LINE_AA)
                    p1, p2 = np.roll(p1, 1), np.roll(p2, 1)
                p1, p2 = np.array([0, v, 0]), np.array([size, v, 0])
                for i in range(3):
                    cv2.line(im, _proj(p1), _proj(p2), (200, 50, 0), 1, cv2.LINE_AA)
                    p1, p2 = np.roll(p1, 1), np.roll(p2, 1)

        super()._draw_on_background(im)

    def _on_input_lines_updated(self) -> None:
        param: LaserParam | None = self.get_laser_param()
        if param is None:
            vec = np.full((4,), np.nan)
            error = np.nan
        else:
            vec = param.vec
            error = param.error
        self.find_component(LabelComponent, "l-param").set_text(
            "\n".join([
                " ".join(f"{col:+9.5f}" for col in vec),
                f"Error: {error}",
            ])
        )

        super()._on_input_lines_updated()

    def _on_button_triggered(self, sender: Component) -> None:
        if sender.get_name() == "b-save":
            param: LaserParam = self.get_laser_param()
            if param is None:
                get_app().make_toast(
                    Toast(
                        self,
                        "error",
                        "Failed to solve laser parameter",
                    )

                )
                return

            def validator(name: str) -> str | None:
                if name == "":
                    return "Please enter a file for this parameter"
                return None

            def already_exist_checker(name: str) -> bool:
                return repo.laser_param.exists(name)

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
                    profile = LaserParamProfile(
                        name=name,
                        param=param,
                    )
                    repo.laser_param.put(profile)
                    get_app().make_toast(
                        Toast(
                            self,
                            "info",
                            f"Saved laser parameter as {name}",
                        )
                    )

            get_app().show_dialog(
                InputNameDialog(
                    title="Save Laser Parameter",
                    validator=validator,
                    already_exist_checker=already_exist_checker,
                    callback=callback,
                )
            )
            return
        if sender.get_name() == "b-back":
            def callback(button_name: str | None) -> None:
                get_app().close_dialog()
                if button_name == "Yes":
                    get_app().move_back()

            n = self._input_lines.get_line_count()
            if n > 0:
                get_app().show_dialog(
                    MessageDialog(
                        is_error=True,
                        message=f"{n} lines are marked. Are you sure to exit?",
                        buttons=("No", "Yes"),
                        callback=callback,
                    )
                )
                return
            else:
                get_app().move_back()

    def _before_add_point(self, x: int, y: int) -> tuple[int, int] | None:
        if self._active_plane_index is None:
            get_app().make_toast(
                Toast(
                    self,
                    "error",
                    "Select a plane in 3D model before you add lines",
                )
            )
            return None

        return x, y

    def _before_add_line(self, x1: int, y1: int, x2: int, y2: int) \
            -> tuple[int, int, int, int, Any] | None:
        return x1, y1, x2, y2, self._active_plane_index

    def mouse_event(self, event: MouseEvent) -> bool:
        if event.left_down:
            # Picture-in-pictureをクリックしていたらクリックした面を記録する
            pos_pip = self.translate_onto_picture_in_picture(event.x, event.y)
            if pos_pip is not None:
                self._active_plane_index = self._calib_model.get_plane_at(
                    int(pos_pip[0]),
                    int(pos_pip[1]),
                )
                return True

        return super().mouse_event(event)

    def key_event(self, event: KeyEvent) -> bool:
        if event.down:
            if event.key == Key.W:
                cb = self.find_component(CheckBoxComponent, "cb-show-3d-model")
                cb.set_value(not cb.get_value())
            if event.key == Key.S:
                cb = self.find_component(CheckBoxComponent, "cb-snap")
                cb.set_value(not cb.get_value())
        return super().key_event(event)
