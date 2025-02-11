from typing import Iterable, Callable

import cv2
import numpy as np
import pandas as pd

import repo.laser_param
from app_logging import create_logger
from core.tk.app import ApplicationWindowSize
from core.tk.color import Color
from core.tk.component.button import ButtonComponent
from core.tk.component.check_box import CheckBoxComponent
from core.tk.component.component import Component
from core.tk.component.label import LabelComponent
from core.tk.component.separator import SeparatorComponent
from core.tk.component.spacer import SpacerComponent
from core.tk.component.toast import Toast
from core.tk.dialog import InputNameDialog, MessageDialog
from core.tk.event import KeyEvent, MouseEvent
from core.tk.global_state import get_app
from core.tk.key import Key
from model.camera_param import CameraParamProfile, CameraParam
from model.image import Image
from model.laser_param import LaserParam, LaserParamProfile
from scene.my_scene import MyScene
from util.camera_calib_model import CameraCalibModel, DEFAULT_CALIB_MODEL
from util.space_alg import Plane3D


class InputLine:
    def __init__(
            self,
            plane_index: int,
            x1: int,
            y1: int,
            x2: int,
            y2: int,
    ):
        self._plane_index = plane_index

        self._x1 = x1
        self._y1 = y1
        self._x2 = x2
        self._y2 = y2
        self._p1 = np.array([self._x1, self._y1], int)
        self._p2 = np.array([self._x2, self._y2], int)
        self._size = np.abs(self._p2 - self._p1)
        self._norm = np.linalg.norm(self._size)

        r = np.linspace(0, 1, 5)[:, None]
        self._points = (self._p1 * r + self._p2 * (1 - r)).round(0).astype(int)
        self._points.setflags(write=False)

    @property
    def start(self) -> tuple[int, int]:
        x, y = self._p1
        return x, y

    @property
    def end(self) -> tuple[int, int]:
        x, y = self._p2
        return x, y

    @property
    def plane_index(self) -> int:
        return self._plane_index

    @property
    def points(self) -> list[tuple[int, int]]:
        if self._points.size == 0:
            return []
        return [(p[0], p[1]) for p in self._points]

    def __len__(self):
        return len(self._points)

    def distance_from_point(self, x, y) -> float:
        p0 = np.array([x, y])
        return np.abs(np.cross(self._p2 - self._p1, p0 - self._p1)) / self._norm


class InputLaserLines:
    _logger = create_logger()

    def __init__(
            self,
            calib_model: CameraCalibModel,
            camera_param: CameraParam,
            callback: Callable[[], None] = None,
    ):
        self._calib_model = calib_model
        self._camera_param = camera_param
        self._callback = callback

        self._lines: list[InputLine] = []

        self._modification_count = 0
        self._laser_param: LaserParam | None = None
        self._laser_param_modification_count = None

    def query_nearest_line(self, x: int, y: int) -> int | None:  # index
        min_distance = float('inf')
        nearest_index = None
        for i, line in enumerate(self._lines):
            distance = line.distance_from_point(x, y)
            if distance < min_distance:
                min_distance = distance
                nearest_index = i
        return nearest_index

    def get_points(self) -> np.ndarray:
        return np.array([p for line in self._lines for p in line.points])

    def get_point_count(self) -> int:
        return len(self.get_points())

    def get_line_count(self) -> int:
        return len(self._lines)

    def get_laser_param(self) -> LaserParam | None:
        if self._laser_param is None \
                or self._laser_param_modification_count != self._modification_count:
            if self.get_point_count() <= 3:
                return None

            points = []
            for line in self._lines:
                plane = Plane3D.from_points(self._calib_model.get_plane_corners(line.plane_index))
                print("plane", plane, self._calib_model.get_plane_corners(line.plane_index))
                for u, v in line.points:
                    x, y, z = self._camera_param.from_2d_to_3d(u, v, plane.a, plane.b, plane.c)
                    print((u, v), (x, y, z))
                    points.append([x, y, z])
            points = np.array(points)

            try:
                p, error = Plane3D.from_points(points, is_complete_plane=False, return_error=True)
                vec = np.array([p.a, p.b, p.c, 1.0])
                self._laser_param = LaserParam(vec=vec, error=error)
            except np.linalg.LinAlgError:
                return None
            self._laser_param_modification_count = self._modification_count
            self._logger.debug(
                f"Camera parameter solved\n"
                f"{pd.DataFrame(self._laser_param.vec).round(6)!s}\n"
                f"{pd.DataFrame(points).round(0)!s}"
            )
        return self._laser_param

    def _dispatch_callback(self):
        if self._callback is not None:
            self._callback()

    def add_line(self, plane_index, x1, y1, x2, y2) -> None:
        self._lines.append(InputLine(plane_index, x1, y1, x2, y2))
        self._modification_count += 1
        self._dispatch_callback()

    def remove_line(self, index: int) -> None:
        del self._lines[index]
        self._modification_count += 1
        self._dispatch_callback()

    def iter_lines(self) -> Iterable[InputLine]:
        yield from self._lines


class LineDrawingState:
    def __init__(self):
        self._p1: tuple[int, int] | None = None
        self._p2: tuple[int, int] | None = None

    def add_point(self, x: int, y: int) -> bool:
        if self._p1 is None:
            self._p1 = (x, y)
            return True
        elif self._p2 is None:
            self._p2 = (x, y)
            return True
        else:
            return False

    def iter_points(self) -> Iterable[tuple[int, int]]:
        if self._p1 is not None:
            yield self._p1
        if self._p2 is not None:
            yield self._p2

    def remove_point(self, x: int, y: int, r: int = 20) -> bool:
        if self._p1 is None:
            distance_1 = np.inf
        else:
            distance_1 = np.linalg.norm([x - self._p1[0], y - self._p1[1]])
            if distance_1 > r:
                distance_1 = np.inf
        if self._p2 is None:
            distance_2 = np.inf
        else:
            distance_2 = np.linalg.norm([x - self._p2[0], y - self._p2[1]])
            if distance_2 > r:
                distance_2 = np.inf
        arg = np.argmin([distance_1, distance_2])
        distance = [distance_1, distance_2][arg]
        if np.isfinite(distance):
            if arg == 0:
                self._p1 = None
            elif arg == 1:
                self._p2 = None
            else:
                assert False, ((x, y), self._p1, self._p2, distance_1, distance_2, arg, distance)
            return True
        else:
            return False

    def is_completed(self) -> bool:
        return self._p1 is not None and self._p2 is not None

    def get_line_and_flush(self) -> tuple[int, int, int, int]:  # x1, y1, x2, y2
        points = self._p1[0], self._p1[1], self._p2[0], self._p2[1]
        self._p1 = None
        self._p2 = None
        return points


class LaserParamScene(MyScene):
    def __init__(self, *, image: Image, camera_param_profile: CameraParamProfile):
        super().__init__()

        self._snap_distance = 20

        self._image: Image = image
        self._camera_param_profile = camera_param_profile

        self._cursor_pos: tuple[int, int] = (0, 0)
        self._active_plane_index: int | None = None

        self._calib_model: CameraCalibModel = DEFAULT_CALIB_MODEL

        self._line_drawing_state = LineDrawingState()

        self._input_lines = InputLaserLines(
            calib_model=self._calib_model,
            camera_param=self._camera_param_profile.param,
            callback=self._on_input_lines_updated,
        )

    def load_event(self):
        self.add_component(LabelComponent(self, "Camera Parameter", bold=True))
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
                "Show 3D model (TAB)",
                True,
                name="cb-show-3d-model",
            )
        )
        self.add_component(SeparatorComponent(self))
        self.add_component(LabelComponent(self, "", name="l-param"))
        self.add_component(ButtonComponent(self, "Save", name="b-save"))
        self.add_component(SpacerComponent(self))
        self.add_component(ButtonComponent(self, "Back", name="b-back"))

    def update(self):
        if self._calib_model is not None \
                and self.find_component(CheckBoxComponent, "cb-show-3d-model").get_value():
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

    def create_background(self, window_size: "ApplicationWindowSize") -> np.ndarray | None:
        canvas = self._image.data.copy()
        cv2.rectangle(
            canvas,
            (0, 0),
            (self._image.data.shape[1], self._image.data.shape[0]),
            (255, 0, 0),
            3,
            cv2.LINE_AA,
        )

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
                    cv2.line(canvas, _proj(p1), _proj(p2), (200, 50, 0), 1, cv2.LINE_AA)
                    p1, p2 = np.roll(p1, 1), np.roll(p2, 1)
                p1, p2 = np.array([0, v, 0]), np.array([size, v, 0])
                for i in range(3):
                    cv2.line(canvas, _proj(p1), _proj(p2), (200, 50, 0), 1, cv2.LINE_AA)
                    p1, p2 = np.roll(p1, 1), np.roll(p2, 1)

        # カーソル
        cv2.circle(
            canvas,
            self._cursor_pos,
            8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.circle(
            canvas,
            self._cursor_pos,
            8,
            (0, 0, 200),
            1,
            cv2.LINE_AA,
        )

        # 点
        handler_radius = 2
        for p in self._line_drawing_state.iter_points():
            cv2.rectangle(
                canvas,
                (p[0] - handler_radius, p[1] - handler_radius),
                (p[0] + handler_radius, p[1] + handler_radius),
                Color.WHITE,
                -1,
                cv2.LINE_AA,
            )
            cv2.rectangle(
                canvas,
                (p[0] - handler_radius, p[1] - handler_radius),
                (p[0] + handler_radius, p[1] + handler_radius),
                Color.BLACK,
                1,
                cv2.LINE_AA,
            )

        # 線
        for line in self._input_lines.iter_lines():
            if line.plane_index == self._active_plane_index:
                color = Color.RED
            else:
                color = Color.WHITE
            edge_color = Color.BLACK
            cv2.line(canvas, line.start, line.end, edge_color, 2, cv2.LINE_AA)
            cv2.line(canvas, line.start, line.end, color, 1, cv2.LINE_AA)
            for p in line.start, line.end:
                cv2.rectangle(
                    canvas,
                    (p[0] - handler_radius, p[1] - handler_radius),
                    (p[0] + handler_radius, p[1] + handler_radius),
                    Color.WHITE,
                    -1,
                    cv2.LINE_AA,
                )
                cv2.rectangle(
                    canvas,
                    (p[0] - handler_radius, p[1] - handler_radius),
                    (p[0] + handler_radius, p[1] + handler_radius),
                    Color.BLACK,
                    1,
                    cv2.LINE_AA,
                )

        return window_size.coerce(canvas)

    def key_event(self, event: KeyEvent) -> bool:
        if event.down:
            if event.key == Key.TAB:
                cb = self.find_component(CheckBoxComponent, "cb-show-3d-model")
                cb.set_value(not cb.get_value())
            if event.key == Key.S:
                cb = self.find_component(CheckBoxComponent, "cb-snap")
                cb.set_value(not cb.get_value())
        return super().key_event(event)

    def mouse_event(self, event: MouseEvent) -> bool:
        if super().mouse_event(event):
            return True
        if event.move:
            self._cursor_pos = event.x, event.y
            return True
        if event.left_down:
            handled = False

            # Picture-in-pictureをクリックしていたらクリックした面を記録する
            pos_pip = self.translate_onto_picture_in_picture(event.x, event.y)
            if pos_pip is not None:
                self._active_plane_index = self._calib_model.get_plane_at(
                    int(pos_pip[0]),
                    int(pos_pip[1]),
                )
                handled = True
            if handled:
                return True

            if self._active_plane_index is None:
                get_app().make_toast(
                    Toast(
                        self,
                        "error",
                        "Select a plane in 3D model before you add lines",
                    )
                )
                return True

            # 線入力のための入力点を追加
            handled = self._line_drawing_state.add_point(event.x, event.y)
            if self._line_drawing_state.is_completed():
                x1, y1, x2, y2 = self._line_drawing_state.get_line_and_flush()
                self._input_lines.add_line(self._active_plane_index, x1, y1, x2, y2)
                return True
            if handled:
                return True

            return handled
        if event.right_down:
            # 線入力のための入力点を削除
            handled = self._line_drawing_state.remove_point(event.x, event.y)
            if handled:
                return True

            # 線を削除
            i = self._input_lines.query_nearest_line(event.x, event.y)
            if i is not None:
                self._input_lines.remove_line(i)
                handled = True
            if handled:
                return True

            return False
        return False

    def _on_input_lines_updated(self) -> None:
        param: LaserParam | None = self._input_lines.get_laser_param()
        if param is None:
            vec = np.full((4,), np.nan)
        else:
            vec = param.vec
        self.find_component(LabelComponent, "l-param").set_text(
            "\n".join([
                " ".join(f"{col:+9.5f}" for col in vec),
                f"Error: {param.error}",
            ])
        )

    def _on_button_triggered(self, sender: Component) -> None:
        if sender.get_name() == "b-save":
            param: LaserParam = self._input_lines.get_laser_param()
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
