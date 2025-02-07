from typing import TYPE_CHECKING

import cv2
import numpy as np

import repo.image
from app_tk.component.button import ButtonComponent
from app_tk.component.check_box import CheckBoxComponent
from app_tk.component.component import Component
from app_tk.component.label import LabelComponent
from app_tk.component.line_edit import LineEditComponent
from app_tk.component.spacer import SpacerComponent
from app_tk.event import KeyEvent, MouseEvent
from app_tk.key import Key
from app_tk.rendering import RenderingContext
from camera_calib_model import CameraCalibModel, DEFAULT_CALIB_MODEL
from dot_snap import DotSnapComputer
from model import Image
from scene_base import MyScene
from scene_select_item import SelectItemScene, SelectItemDelegate

if TYPE_CHECKING:
    from app_tk.app import Application


class SelectImageScene(SelectItemScene):
    def render_canvas(self) -> np.ndarray | None:
        return None

    def selection_change_event(self, name: str | None):
        super().selection_change_event(name)
        if name is not None:
            self.set_picture_in_picture(repo.image.get(name).data)


class CameraParaSelectImageDelegate(SelectItemDelegate):
    def __init__(self, scene: "CameraParamScene"):
        self._scene = scene

    def list_name(self) -> list[str]:
        return repo.image.list_names()

    def execute(self, name: str) -> str | None:
        image = repo.image.get(name)
        self._scene.set_target_image(image)
        return None


class CameraParamScene(MyScene):
    def __init__(self, app: "Application"):
        super().__init__(app)

        self._image: Image | None = None
        self._dot_snap_computer: DotSnapComputer | None = None

        self._cursor_pos: tuple[int, int] | None = None

        self._active_world_point_index = 0

        self._calib_model: CameraCalibModel | None = None
        self.set_calib_model(DEFAULT_CALIB_MODEL)

        self._show_model = True

        self._points: dict[tuple[float, float, float], tuple[int, int]] = {}

    def set_target_image(self, image: Image) -> None:
        self._image = image
        self._dot_snap_computer = DotSnapComputer(
            cv2.cvtColor(self._image.data, cv2.COLOR_BGR2GRAY),
            crop_radius=20,
            min_samples=5,
            snap_radius=30,
            stride=5,
        )
        self.find_component(LabelComponent, "l-image-name").set_text(image.name)

    def set_calib_model(self, calib_model: CameraCalibModel):
        self._calib_model = calib_model

    def load_event(self):
        self.add_component(LabelComponent, "Camera Parameter", bold=True)
        self.add_component(SpacerComponent)
        self.add_component(ButtonComponent, "Select Image", name="b-select-image")
        self.add_component(LabelComponent, "", name="l-image-name")
        self.add_component(SpacerComponent)
        self.add_component(CheckBoxComponent, "Snap to auto detected points", True, name="cb-snap")
        self.add_component(LabelComponent, "Press <TAB> to hide model")
        self.add_component(SpacerComponent)
        self.add_component(ButtonComponent, "Back", name="b-back")

    def update(self):
        if self._calib_model is not None and self._image is not None and self._show_model:
            self.set_picture_in_picture(
                self._calib_model.render_3d(500, 500, self._active_world_point_index),
                500,
            )
        else:
            self.set_picture_in_picture(None)
        return super().update()

    def render_canvas(self) -> np.ndarray | None:
        if self._image is None:
            return None

        canvas = self._image.data.copy()
        cv2.rectangle(
            canvas,
            (0, 0),
            (self._image.data.shape[1], self._image.data.shape[0]),
            (255, 0, 0),
            3,
            cv2.LINE_AA,
        )

        if self._cursor_pos is not None:
            cv2.circle(
                canvas,
                self._cursor_pos,
                8,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        for p2d in self._points.values():
            cv2.circle(
                canvas,
                p2d,
                2,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        return canvas

    def key_event(self, event: KeyEvent) -> bool:
        if event.down:
            if event.key == Key.TAB:
                self._show_model = not self._show_model
        return super().key_event(event)

    def mouse_event(self, event: MouseEvent) -> bool:
        if super().mouse_event(event):
            return True
        if self._image is not None:
            if event.move:
                if self.find_component(CheckBoxComponent, "cb-snap").get_value():
                    snap_pos = self._dot_snap_computer.find_snap_pos((event.x, event.y))
                    if snap_pos is not None:
                        self._cursor_pos = snap_pos
                    else:
                        self._cursor_pos = (event.x, event.y)
                else:
                    self._cursor_pos = (event.x, event.y)
                return True
            if event.left_down:
                if self._cursor_pos is not None:
                    p3d = self._calib_model.get_world_point(self._active_world_point_index)
                    p2d = self._cursor_pos
                    self._points[p3d] = p2d
                    self._active_world_point_index += 1
                    self._active_world_point_index %= self._calib_model.get_world_point_count()
        return False

    def set_overwrite_state(self, state: bool):
        if state:
            # already exists
            self.find_component(ButtonComponent, "b-save").set_text("Overwrite and Save")
            self.find_component(LabelComponent, "l-info").set_text(
                f"Name already exists. Are you sure to overwrite it?"
            )
        else:
            self.find_component(ButtonComponent, "b-save").set_text("Save")
            self.find_component(LabelComponent, "l-info").set_text("")

    def on_value_changed(self, sender: Component) -> None:
        if sender.get_name() == "e-name":
            name = self.find_component(LineEditComponent, "e-name").get_value()
            self.set_overwrite_state(repo.image.exists(name))
            return
        super().on_value_changed(sender)

    def on_button_triggered(self, sender: Component) -> None:
        if sender.get_name() == "b-select-image":
            self.get_app().move_to(
                SelectImageScene(self.get_app(), CameraParaSelectImageDelegate(self))
            )
            return
        if sender.get_name() == "b-back":
            self.get_app().go_back()
