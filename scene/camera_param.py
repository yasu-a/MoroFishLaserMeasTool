from typing import Iterable, Callable

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

import repo.camera_param
from app_logging import create_logger
from core.tk.app import ApplicationWindowSize
from core.tk.component.button import ButtonComponent
from core.tk.component.check_box import CheckBoxComponent
from core.tk.component.component import Component
from core.tk.component.label import LabelComponent
from core.tk.component.separator import SeparatorComponent
from core.tk.component.spacer import SpacerComponent
from core.tk.component.spin_box import SpinBoxComponent
from core.tk.component.toast import Toast
from core.tk.dialog import InputNameDialog, MessageDialog
from core.tk.event import KeyEvent, MouseEvent
from core.tk.global_state import get_app
from core.tk.key import Key
from dot_snap import DotSnapComputer
from model.camera_param import CameraParamProfile, CameraParam
from model.image import Image
from scene.my_scene import MyScene
from util.camera_calib_model import CameraCalibModel, DEFAULT_CALIB_MODEL
from util.solve_parameter import solve_equations_camera


class InputPoints:
    _logger = create_logger()

    def __init__(self, calib_model: CameraCalibModel, callback: Callable[[], None] = None):
        self._calib_model = calib_model
        self._callback = callback

        self._points_2d: list[tuple[int, int] | None] \
            = [None] * calib_model.get_world_point_count()
        self._points_2d_kdtree: KDTree | None = None  # None if empty
        self._kdtree_index_mapping: dict[int, int] = {}

        self._modification_count = 0
        self._camera_param = None
        self._camera_param_modification_count = None

    def get_camera_mat(self) -> np.ndarray | None:  # TODO: 廃止
        if self._camera_param is None \
                or self._camera_param_modification_count != self._modification_count:
            if self.get_input_point_count() <= 1:
                return None
            points = []
            for i in self.iter_indexes():
                if not self.is_marked(i):
                    continue
                points.append([*self._calib_model.get_world_point(i), *self.point_at(i)])
            points = np.array(points)
            try:
                self._camera_param = solve_equations_camera(points)
            except np.linalg.LinAlgError:
                return None
            self._camera_param_modification_count = self._modification_count
            self._logger.debug(
                f"Camera parameter solved\n"
                f"{pd.DataFrame(self._camera_param).round(6)!s}\n"
                f"{pd.DataFrame(points).round(0)!s}"
            )
        return self._camera_param

    def get_camera_param(self) -> CameraParam | None:
        mat = self.get_camera_mat()
        if mat is None:
            return None
        return CameraParam(
            mat=mat,
        )

    def _update_kdtree(self):
        available_points_2d = [
            p
            for p in self._points_2d
            if p is not None
        ]

        self._kdtree_index_mapping = {}
        j = 0
        for i in range(len(self._points_2d)):
            if self._points_2d[i] is not None:
                self._kdtree_index_mapping[j] = i
                j += 1

        if not available_points_2d:
            self._points_2d_kdtree = None
            return
        self._points_2d_kdtree = KDTree(available_points_2d)

    def _dispatch_callback(self):
        if self._callback is not None:
            self._callback()

    def add_point(self, index: int, p2d: tuple[int, int]) -> None:
        self._points_2d[index] = p2d
        self._update_kdtree()
        self._modification_count += 1
        self._dispatch_callback()

    def remove_point(self, index: int) -> None:
        self._points_2d[index] = None
        self._update_kdtree()
        self._modification_count += 1
        self._dispatch_callback()

    def query_nearest_index(self, x: int, y: int, r: float) \
            -> tuple[int | None, float | None]:  # index and distance
        if self._points_2d_kdtree is None:
            return None, np.inf
        d, i = self._points_2d_kdtree.query([x, y])
        i = self._kdtree_index_mapping[i]
        if d > r:
            return None, np.inf
        return i, d

    def is_marked(self, index: int) -> bool:
        return self._points_2d[index] is not None

    def point_at(self, index: int) -> tuple[int, int]:
        p = self._points_2d[index]
        assert p is not None
        return p

    def find_point(self, point: tuple[int, int]) -> int | None:
        for i, p in enumerate(self._points_2d):
            if p is None:
                continue
            if p == point:
                return i
        return None

    def iter_indexes(self) -> Iterable[int]:
        yield from range(len(self._points_2d))

    def get_input_point_count(self) -> int:
        return sum(1 for i in self.iter_indexes() if self.is_marked(i))


class SnapManager:
    def __init__(self, input_points: InputPoints, dot_snap_points: list[tuple[int, int]]):
        self._input_points = input_points
        self._dot_snap_points = dot_snap_points
        self._dot_snap_points_kdtree = KDTree(dot_snap_points)

    def query_nearest_snap(self, x: int, y: int, r: float) -> tuple[int, int] | None:
        i_input, d_input = self._input_points.query_nearest_index(x, y, r)

        d_dot_snap, i_dot_snap = self._dot_snap_points_kdtree.query([x, y])
        if d_dot_snap > r:
            i_dot_snap, d_dot_snap = None, np.inf

        if i_input is None and i_dot_snap is None:
            return None

        if d_input < d_dot_snap:
            return self._input_points.point_at(i_input)
        else:
            return self._dot_snap_points[i_dot_snap]


class CameraParamScene(MyScene):
    def __init__(self, *, image: Image):
        super().__init__()

        self._snap_distance = 20

        self._image: Image = image
        dot_snap_computer: DotSnapComputer = DotSnapComputer(
            cv2.cvtColor(self._image.data, cv2.COLOR_BGR2GRAY),
            crop_radius=20,
            min_samples=5,
            snap_radius=30,
            stride=5,
        )

        self._cursor_pos: tuple[int, int] = (0, 0)

        self._calib_model: CameraCalibModel = DEFAULT_CALIB_MODEL

        self._input_points = InputPoints(
            calib_model=self._calib_model,
            callback=self._on_input_points_updated,
        )
        self._snap_manager = SnapManager(
            input_points=self._input_points,
            dot_snap_points=dot_snap_computer.compute_snap_positions().tolist(),
        )

        self._active_plane_index: int | None = None

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
                "Show 3D model (W)",
                True,
                name="cb-show-3d-model",
            )
        )
        self.add_component(SpacerComponent(self))
        self.add_component(LabelComponent(self, "World point: (A) <-> (D)"))
        self.add_component(
            SpinBoxComponent(
                self,
                min_value=0,
                max_value=self._calib_model.get_world_point_count() - 1,
                name="sb-active-point-index",
            )
        )
        self.add_component(SeparatorComponent(self))
        self.add_component(LabelComponent(self, "", name="l-param"))
        self.add_component(ButtonComponent(self, "Save", name="b-save"))
        self.add_component(SpacerComponent(self))
        self.add_component(ButtonComponent(self, "Back", name="b-back"))

    def get_active_point_index(self) -> int:
        sb = self.find_component(SpinBoxComponent, "sb-active-point-index")
        return sb.get_value()

    def update(self):
        if self._calib_model is not None \
                and self.find_component(CheckBoxComponent, "cb-show-3d-model").get_value():
            p_highlighted = {}
            for i in self._input_points.iter_indexes():
                if self._input_points.is_marked(i):
                    p_highlighted[i] = 0, 200, 0
            p_highlighted[self.get_active_point_index()] = 0, 0, 255

            plane_colors = {}
            if self._active_plane_index is not None:
                plane_colors[self._active_plane_index] = 255, 0, 0

            self.set_picture_in_picture(
                self._calib_model.render_3d(
                    width=500,
                    height=500,
                    point_colors=p_highlighted,
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

        mat: np.ndarray | None = self._input_points.get_camera_mat()
        if mat is not None:
            def _proj(p):
                _p3d = np.array([*p, 1])
                _p2d = mat @ _p3d
                _p2d /= _p2d[2]
                return _p2d[:2].round(0).astype(int)

            # カメラパラメータ行列が張る空間の格子線の描画
            size = self._calib_model.get_size()
            if self._active_plane_index is None:
                for v in np.arange(0, size + 1e-6, 20):
                    p1, p2 = np.array([v, 0, 0]), np.array([v, size, 0])
                    for i in range(3):
                        cv2.line(canvas, _proj(p1), _proj(p2), (200, 50, 0), 1, cv2.LINE_AA)
                        p1, p2 = np.roll(p1, 1), np.roll(p2, 1)
                    p1, p2 = np.array([0, v, 0]), np.array([size, v, 0])
                    for i in range(3):
                        cv2.line(canvas, _proj(p1), _proj(p2), (200, 50, 0), 1, cv2.LINE_AA)
                        p1, p2 = np.roll(p1, 1), np.roll(p2, 1)
            else:
                active_plane_corners = self._calib_model.get_plane_corners(self._active_plane_index)
                # x plane
                x = active_plane_corners[0, 0]
                if np.all(x == active_plane_corners[1:, 0]):
                    for v in np.arange(0, size + 1e-6, 20):
                        p1, p2 = np.array([x, v, 0]), np.array([x, v, size])
                        cv2.line(canvas, _proj(p1), _proj(p2), (200, 50, 0), 1, cv2.LINE_AA)
                        p1, p2 = np.array([x, 0, v]), np.array([x, size, v])
                        cv2.line(canvas, _proj(p1), _proj(p2), (200, 50, 0), 1, cv2.LINE_AA)
                # y plane
                y = active_plane_corners[0, 1]
                if np.all(y == active_plane_corners[1:, 1]):
                    for v in np.arange(0, size + 1e-6, 20):
                        p1, p2 = np.array([v, y, 0]), np.array([v, y, size])
                        cv2.line(canvas, _proj(p1), _proj(p2), (200, 50, 0), 1, cv2.LINE_AA)
                        p1, p2 = np.array([0, y, v]), np.array([size, y, v])
                        cv2.line(canvas, _proj(p1), _proj(p2), (200, 50, 0), 1, cv2.LINE_AA)
                # z plane
                z = active_plane_corners[0, 2]
                if np.all(z == active_plane_corners[1:, 2]):
                    for v in np.arange(0, size + 1e-6, 20):
                        p1, p2 = np.array([v, 0, z]), np.array([v, size, z])
                        cv2.line(canvas, _proj(p1), _proj(p2), (200, 50, 0), 1, cv2.LINE_AA)
                        p1, p2 = np.array([0, v, z]), np.array([size, v, z])
                        cv2.line(canvas, _proj(p1), _proj(p2), (200, 50, 0), 1, cv2.LINE_AA)

            # 点予測
            p2d = _proj(self._calib_model.get_world_point(self.get_active_point_index()))
            cv2.circle(
                canvas,
                p2d,
                20,
                (50, 50, 50),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "PREDICTION",
                p2d + [20, 20],
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (50, 50, 50),
                1,
                cv2.LINE_AA,
            )

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
        for i in (self._input_points.iter_indexes()):
            if not self._input_points.is_marked(i):
                continue
            p2d = self._input_points.point_at(i)
            cv2.circle(
                canvas,
                p2d,
                2,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.circle(
                canvas,
                p2d,
                2,
                (0, 0, 200) if i == self.get_active_point_index() else (0, 200, 0),
                2,
                cv2.LINE_AA,
            )

        return window_size.coerce(canvas)

    def move_onto_prev_point(self):
        sb = self.find_component(SpinBoxComponent, "sb-active-point-index")
        sb.set_value((sb.get_value() - 1) % self._calib_model.get_world_point_count())

    def move_onto_next_point(self):
        sb = self.find_component(SpinBoxComponent, "sb-active-point-index")
        sb.set_value((sb.get_value() + 1) % self._calib_model.get_world_point_count())

    def key_event(self, event: KeyEvent) -> bool:
        if event.down:
            if event.key == Key.W:
                cb = self.find_component(CheckBoxComponent, "cb-show-3d-model")
                cb.set_value(not cb.get_value())
            if event.key == Key.S:
                cb = self.find_component(CheckBoxComponent, "cb-snap")
                cb.set_value(not cb.get_value())
        if event.enter:
            if event.key == Key.A:
                self.move_onto_prev_point()
            if event.key == Key.D:
                self.move_onto_next_point()
        return super().key_event(event)

    def mouse_event(self, event: MouseEvent) -> bool:
        if super().mouse_event(event):
            return True
        if event.move:
            if self.find_component(CheckBoxComponent, "cb-snap").get_value():
                snap_pos = self._snap_manager.query_nearest_snap(
                    event.x, event.y, self._snap_distance
                )
                if snap_pos is not None:
                    self._cursor_pos = snap_pos
                else:
                    self._cursor_pos = (event.x, event.y)
            else:
                self._cursor_pos = (event.x, event.y)
            return True
        if event.left_down:
            # Picture-in-pictureをクリックしていたらクリックした面を記録する
            pos_pip = self.translate_onto_picture_in_picture(event.x, event.y)
            if pos_pip is not None:
                clicked_plane_index = self._calib_model.get_plane_at(
                    int(pos_pip[0]),
                    int(pos_pip[1]),
                )
                if self._active_plane_index == clicked_plane_index:
                    self._active_plane_index = None
                else:
                    self._active_plane_index = clicked_plane_index
                return True

            # 点を追加する
            p2d = self._cursor_pos
            self._input_points.add_point(self.get_active_point_index(), p2d)
            self.move_onto_next_point()
        if event.right_down:
            snap_pos = self._snap_manager.query_nearest_snap(
                event.x, event.y, self._snap_distance
            )
            if snap_pos is not None:
                index = self._input_points.find_point(self._cursor_pos)
                if index is not None:
                    self._input_points.remove_point(index)
                    return True

        return False

    def _on_input_points_updated(self) -> None:
        mat: np.ndarray | None = self._input_points.get_camera_mat()
        if mat is None:
            mat = np.full((3, 4), np.nan)
        self.find_component(LabelComponent, "l-param").set_text(
            "\n".join(
                " ".join(f"{col:+9.5f}" for col in row)
                for row in mat
            )
        )

    def _on_button_triggered(self, sender: Component) -> None:
        if sender.get_name() == "b-save":
            mat: np.ndarray | None = self._input_points.get_camera_mat()
            if mat is None:
                get_app().make_toast(
                    Toast(
                        self,
                        "error",
                        "Failed to solve camera parameter",
                    )

                )
                return

            def validator(name: str) -> str | None:
                if name == "":
                    return "Please enter a file for this parameter"
                if name.strip() != name:
                    return "File name cannot contain leading or trailing spaces"
                return None

            def already_exist_checker(name: str) -> bool:
                return repo.camera_param.exists(name)

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
                    profile = CameraParamProfile(
                        name=name,
                        param=CameraParam(
                            mat=mat,
                        ),
                    )
                    repo.camera_param.put(profile)
                    get_app().make_toast(
                        Toast(
                            self,
                            "info",
                            f"Saved camera parameter as {name}",
                        )
                    )

            get_app().show_dialog(
                InputNameDialog(
                    title="Save Camera Parameter",
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

            n = self._input_points.get_input_point_count()
            if n > 0:
                get_app().show_dialog(
                    MessageDialog(
                        is_error=True,
                        message=f"{n} points are marked. Are you sure to exit?",
                        buttons=("No", "Yes"),
                        callback=callback,
                    )
                )
                return
            else:
                get_app().move_back()
