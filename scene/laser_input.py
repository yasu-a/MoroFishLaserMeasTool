import copy
from abc import abstractmethod
from typing import Callable, Iterable, Any

import cv2
import numpy as np

from app_logging import create_logger
from core.tk.app import ApplicationWindowSize
from core.tk.color import Color
from core.tk.component.toast import Toast
from core.tk.event import KeyEvent, MouseEvent
from core.tk.global_state import get_app
from model.image import Image
from model.laser_param import LaserParam
from scene.my_scene import MyScene


class InputLine:
    def __init__(
            self,
            x1: int,
            y1: int,
            x2: int,
            y2: int,
            *,
            user_data: Any | None = None,
            n_sample_points: int,
    ):
        self._user_data = user_data

        self._x1 = x1
        self._y1 = y1
        self._x2 = x2
        self._y2 = y2
        self._p1 = np.array([self._x1, self._y1], int)
        self._p2 = np.array([self._x2, self._y2], int)
        self._size = np.abs(self._p2 - self._p1)
        self._norm = np.linalg.norm(self._size)

        # sample points
        r = np.linspace(0, 1, n_sample_points)[:, None]
        points = set(map(tuple, (self._p1 * r + self._p2 * (1 - r)).round(0).astype(int)))
        self._points: list[tuple[int, int]] = [(int(u), int(v)) for u, v in points]

    @property
    def start(self) -> tuple[int, int]:
        x, y = self._p1
        return x, y

    @property
    def end(self) -> tuple[int, int]:
        x, y = self._p2
        return x, y

    @property
    def user_data(self) -> Any:
        return self._user_data

    @property
    def points(self) -> list[tuple[int, int]]:
        return copy.copy(self._points)

    def __len__(self):
        return len(self._points)

    def distance_from_point(self, x, y) -> float:
        p0 = np.array([x, y])
        return np.abs(np.cross(self._p2 - self._p1, p0 - self._p1)) / self._norm


class InputLaserLines:
    _logger = create_logger()

    def __init__(
            self,
            callback: Callable[[], None] = None,
            *,
            n_sample_points_per_line: int,
    ):
        self._callback = callback
        self._n_sample_points_per_line = n_sample_points_per_line

        self._lines: list[InputLine] = []

        self._modification_count = 0
        self._laser_param: LaserParam | None = None
        self._laser_param_modification_count = None

    def query_nearest_line(self, x: int, y: int, r: int) -> int | None:  # index
        min_distance = r
        nearest_index = None
        for i, line in enumerate(self._lines):
            distance = line.distance_from_point(x, y)
            if distance < min_distance:
                min_distance = distance
                nearest_index = i
        return nearest_index

    def iter_points(self) -> Iterable[tuple[int, int]]:
        for line in self._lines:
            for p in line.points:
                yield p

    def get_point_count(self) -> int:
        return sum(len(line.points) for line in self._lines)

    def get_line_count(self) -> int:
        return len(self._lines)

    def _dispatch_callback(self):
        if self._callback is not None:
            self._callback()

    def add_line(self, x1, y1, x2, y2, *, user_data: Any = None) -> None:
        self._lines.append(
            InputLine(
                x1, y1, x2, y2,
                user_data=user_data,
                n_sample_points=self._n_sample_points_per_line,
            )
        )
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


class SnapManager:
    def __init__(
            self,
            *,
            snap_radius: int,
            snap_point_iterator_producer: Callable[[], Iterable[tuple[int, int]]],
    ):
        self._snap_radius = snap_radius
        self._snap_point_iterator_producer = snap_point_iterator_producer

    def get_snap_point(self, x: int, y: int) -> tuple[int, int] | None:
        distances = []
        points = []
        for u, v in self._snap_point_iterator_producer():
            points.append((u, v))
            d = np.linalg.norm([x - u, y - v])
            if d > self._snap_radius:
                d = np.inf
            distances.append(d)

        if len(distances) == 0:
            return None

        i = np.argmin(distances)
        if not np.isfinite(distances[i]):
            return None

        p = points[i]
        return p


class LaserInputScene(MyScene):
    def __init__(self, *, image: Image, n_sample_points_per_line):
        super().__init__()

        self._snap_distance = 20

        self._image: Image = image

        self._cursor_pos: tuple[int, int] = (0, 0)

        self._line_drawing_state = LineDrawingState()

        self._snap_manager = SnapManager(
            snap_radius=30,
            snap_point_iterator_producer=self._iter_snap_points,
        )

        self._input_lines = InputLaserLines(
            callback=self._on_input_lines_updated,
            n_sample_points_per_line=n_sample_points_per_line,
        )

    @abstractmethod
    def _is_snap_enabled(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def _is_show_3d_model_enabled(self) -> bool:
        raise NotImplementedError()

    def _iter_snap_points(self) -> Iterable[tuple[int, int]]:
        yield from self._input_lines.iter_points()

    def _get_line_count(self) -> int:  # TODO: _input_linesへのアクセスをこれに置き換える
        return self._input_lines.get_line_count()

    def _get_point_count(self) -> int:  # TODO: _input_linesへのアクセスをこれに置き換える
        return self._input_lines.get_point_count()

    @abstractmethod
    def _get_line_color(self, line: InputLine) -> Color:
        raise NotImplementedError()

    def _draw_on_background(self, im: np.ndarray) -> None:
        # カーソル
        cv2.circle(
            im,
            self._cursor_pos,
            8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.circle(
            im,
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
                im,
                (p[0] - handler_radius, p[1] - handler_radius),
                (p[0] + handler_radius, p[1] + handler_radius),
                Color.WHITE,
                -1,
                cv2.LINE_AA,
            )
            cv2.rectangle(
                im,
                (p[0] - handler_radius, p[1] - handler_radius),
                (p[0] + handler_radius, p[1] + handler_radius),
                Color.BLACK,
                1,
                cv2.LINE_AA,
            )

        # 線
        for line in self._input_lines.iter_lines():
            color = self._get_line_color(line)
            edge_color = Color.BLACK
            cv2.line(im, line.start, line.end, edge_color, 2, cv2.LINE_AA)
            cv2.line(im, line.start, line.end, color, 1, cv2.LINE_AA)
            for p in line.start, line.end:
                cv2.rectangle(
                    im,
                    (p[0] - handler_radius, p[1] - handler_radius),
                    (p[0] + handler_radius, p[1] + handler_radius),
                    Color.WHITE,
                    -1,
                    cv2.LINE_AA,
                )
                cv2.rectangle(
                    im,
                    (p[0] - handler_radius, p[1] - handler_radius),
                    (p[0] + handler_radius, p[1] + handler_radius),
                    Color.BLACK,
                    1,
                    cv2.LINE_AA,
                )

    def create_background(self, window_size: "ApplicationWindowSize") -> np.ndarray | None:
        im = self._image.data.copy()
        cv2.rectangle(
            im,
            (0, 0),
            (self._image.data.shape[1], self._image.data.shape[0]),
            (255, 0, 0),
            3,
            cv2.LINE_AA,
        )

        self._draw_on_background(im)

        return window_size.coerce(im)

    def key_event(self, event: KeyEvent) -> bool:
        return super().key_event(event)

    @abstractmethod
    def _before_add_point(self, x: int, y: int) -> tuple[int, int] | None:
        # returns x, y
        # returns None if cancel the event
        raise NotImplementedError()

    @abstractmethod
    def _before_add_line(self, x1: int, y1: int, x2: int, y2: int) \
            -> tuple[int, int, int, int, Any] | None:
        # returns x1, y1, x2, y2, user_data
        # returns None if cancel the event
        raise NotImplementedError()

    def mouse_event(self, event: MouseEvent) -> bool:
        if event.move:
            snap_pos = None
            if self._is_snap_enabled():
                snap_pos = self._snap_manager.get_snap_point(event.x, event.y)
            self._cursor_pos = snap_pos or (event.x, event.y)
            return True

        if event.left_down:
            # 線入力のための入力点を追加
            x, y = self._cursor_pos
            result = self._before_add_point(x, y)
            if result is not None:
                x, y = result
                is_point_added = self._line_drawing_state.add_point(x, y)
            else:
                is_point_added = False
            if is_point_added:
                # 線ができたら線を入力
                if self._line_drawing_state.is_completed():
                    x1, y1, x2, y2 = self._line_drawing_state.get_line_and_flush()
                    if (x1, y1) == (x2, y2):
                        get_app().make_toast(
                            Toast(
                                self,
                                "error",
                                "You need two unique points to make a line",
                            )
                        )
                    else:
                        result = self._before_add_line(x1, y1, x2, y2)
                        if result is not None:
                            x1, y1, x2, y2, user_data = result
                            self._input_lines.add_line(x1, y1, x2, y2, user_data=user_data)
                return True

            return False
        if event.right_down:
            # 線入力のための入力点を削除
            handled = self._line_drawing_state.remove_point(*self._cursor_pos)
            if handled:
                return True

            # 線を削除
            i = self._input_lines.query_nearest_line(*self._cursor_pos, r=30)
            if i is not None:
                self._input_lines.remove_line(i)
                handled = True
            if handled:
                return True

            return False

        return super().mouse_event(event)

    def _on_input_lines_updated(self) -> None:
        pass
