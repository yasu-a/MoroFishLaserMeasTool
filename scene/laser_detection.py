from typing import Any

import cv2
import numpy as np
import pandas as pd

import repo.laser_detection
from app_logging import create_logger
from core.tk.color import Color
from core.tk.component.button import ButtonComponent
from core.tk.component.check_box import CheckBoxComponent
from core.tk.component.component import Component
from core.tk.component.label import LabelComponent
from core.tk.component.separator import SeparatorComponent
from core.tk.component.spacer import SpacerComponent
from core.tk.component.spin_box import SpinBoxComponent
from core.tk.component.toast import Toast
from core.tk.dialog import InputNameDialog, MessageDialog
from core.tk.event import KeyEvent
from core.tk.global_state import get_app
from core.tk.key import Key
from model.image import Image
from model.laser_detection import LaserDetectionProfile, LaserDetectionModel
from scene.laser_input import LaserInputScene, InputLine


class LaserDetectionScene(LaserInputScene):
    _logger = create_logger()

    def __init__(self, *, image: Image):
        super().__init__(image=image, n_sample_points_per_line=100)

        self._im_hsv = cv2.cvtColor(image.data, cv2.COLOR_BGR2HSV)

        self._model_cache_hash: int | None = None
        self._model_cache: LaserDetectionModel | None = None

    def load_event(self):
        self.add_component(LabelComponent(self, "Laser Detection Model", bold=True))

        self.add_component(SeparatorComponent(self))

        self.add_component(
            LabelComponent(self, " ", name="l-info")
        )

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
        self.add_component(
            CheckBoxComponent(
                self,
                "Show Mask",
                True,
                name="cb-show-mask",
            )
        )
        self.add_component(LabelComponent(self, "HSV Min"))
        for c in "hsv":
            self.add_component(
                SpinBoxComponent(
                    self,
                    value=0,
                    min_value=0,
                    max_value=255,
                    name=f"sb-{c}-min",
                )
            )
        self.add_component(LabelComponent(self, "HSV Max"))
        for c in "hsv":
            self.add_component(
                SpinBoxComponent(
                    self,
                    value=0,
                    min_value=0,
                    max_value=255,
                    name=f"sb-{c}-max",
                )
            )
        self.add_component(LabelComponent(self, "Morphology Opening"))
        self.add_component(
            SpinBoxComponent(
                self,
                value=1,
                min_value=0,
                max_value=50,
                name=f"sb-morph-open",
            )
        )
        self.add_component(LabelComponent(self, "Morphology Closing"))
        self.add_component(
            SpinBoxComponent(
                self,
                value=1,
                min_value=0,
                max_value=50,
                name=f"sb-morph-close",
            )
        )

        self.add_component(SeparatorComponent(self))

        self.add_component(LabelComponent(self, "", name="l-model"))
        self.add_component(ButtonComponent(self, "Save", name="b-save"))
        self.add_component(ButtonComponent(self, "Clear", name="b-clear"))

        self.add_component(SpacerComponent(self))

        self.add_component(ButtonComponent(self, "Back", name="b-back"))

    def _is_snap_enabled(self) -> bool:
        return self.find_component(CheckBoxComponent, "cb-snap").get_value()

    def _is_show_3d_model_enabled(self) -> bool:
        return self.find_component(CheckBoxComponent, "cb-show-3d-model").get_value()

    def _get_morph_open_size(self) -> int:
        return self.find_component(SpinBoxComponent, "sb-morph-open").get_value()

    def _set_morph_open_size(self, value: int) -> None:
        self.find_component(SpinBoxComponent, "sb-morph-open").set_value(value)

    def _get_morph_close_size(self) -> int:
        return self.find_component(SpinBoxComponent, "sb-morph-close").get_value()

    def _set_morph_close_size(self, value: int) -> None:
        self.find_component(SpinBoxComponent, "sb-morph-close").set_value(value)

    def _get_hsv_min(self) -> tuple[int, int, int]:
        return (
            self.find_component(SpinBoxComponent, f"sb-h-min").get_value(),
            self.find_component(SpinBoxComponent, f"sb-s-min").get_value(),
            self.find_component(SpinBoxComponent, f"sb-v-min").get_value(),
        )

    def _set_hsv_min(self, hsv: tuple[int, int, int]) -> None:
        self.find_component(SpinBoxComponent, f"sb-h-min").set_value(hsv[0])
        self.find_component(SpinBoxComponent, f"sb-s-min").set_value(hsv[1])
        self.find_component(SpinBoxComponent, f"sb-v-min").set_value(hsv[2])

    def _get_hsv_max(self) -> tuple[int, int, int]:
        return (
            self.find_component(SpinBoxComponent, f"sb-h-max").get_value(),
            self.find_component(SpinBoxComponent, f"sb-s-max").get_value(),
            self.find_component(SpinBoxComponent, f"sb-v-max").get_value(),
        )

    def _set_hsv_max(self, hsv: tuple[int, int, int]) -> None:
        self.find_component(SpinBoxComponent, f"sb-h-max").set_value(hsv[0])
        self.find_component(SpinBoxComponent, f"sb-s-max").set_value(hsv[1])
        self.find_component(SpinBoxComponent, f"sb-v-max").set_value(hsv[2])

    def _clear_model(self) -> None:
        self._set_hsv_min((0, 0, 0))
        self._set_hsv_max((0, 0, 0))
        self._set_morph_open_size(1)
        self._set_morph_close_size(1)

    def update(self):
        info_label = self.find_component(LabelComponent, "l-info")
        x, y = self._cursor_pos
        try:
            h, s, v = self._im_hsv[y, x]
            info_label.set_text(f"H: {h:3d}, S: {s:3d}, V: {v:3d}")
        except IndexError:
            info_label.set_text(" ")
        super().update()

    def get_laser_detection_model(self) -> LaserDetectionModel | None:
        def source_hash() -> int:
            keys = (
                self._get_morph_close_size(),
                self._get_morph_open_size(),
                self._get_hsv_min(),
                self._get_hsv_max(),
            )
            return hash(keys)

        current_hash = source_hash()
        if self._model_cache_hash != current_hash or self._model_cache is None:
            self._model_cache_hash = current_hash

            model = LaserDetectionModel(
                hsv_min=np.array(self._get_hsv_min(), np.uint8),
                hsv_max=np.array(self._get_hsv_max(), np.uint8),
                morph_open_size=self._get_morph_open_size(),
                morph_close_size=self._get_morph_close_size(),
            )
            self._logger.debug(
                f"Laser detection model determined\n"
                f"{pd.DataFrame(model.hsv_min).round(6)!s}\n"
                f"{pd.DataFrame(model.hsv_max).round(6)!s}\n"
            )
            self._model_cache = model

        return self._model_cache

    def _get_line_color(self, line: InputLine) -> Color:
        return Color.WHITE

    def _draw_on_background(self, im: np.ndarray) -> None:
        if self.find_component(CheckBoxComponent, "cb-show-mask").get_value():
            model = self.get_laser_detection_model()
            if model is not None:
                mask = model.create_laser_mask(self._im_hsv, is_hsv=True)
                im[mask.astype(bool)] = np.array([0, 0, 255])
        super()._draw_on_background(im)

    def _on_input_lines_updated(self) -> None:
        if self._get_point_count() > 3:
            points = np.array(list(set(self._input_lines.iter_points())))
            colors = self._im_hsv[points[:, 1], points[:, 0], :]
            hsv_min, hsv_max = np.percentile(colors, [0, 100], axis=0).round(0).astype(np.uint8)
            self._set_hsv_min(hsv_min)
            self._set_hsv_max(hsv_max)
        else:
            hsv_min = np.full(3, np.nan)
            hsv_max = np.full(3, np.nan)

        self.find_component(LabelComponent, "l-model").set_text(
            "\n".join([
                " ".join([" " * 4, "H".rjust(4), "S".rjust(4), "V".rjust(4)]),
                " ".join([" " * 4, *(f"{v:>4.0f}" for v in hsv_min)]),
                " ".join([" " * 4, *(f"{v:>4.0f}" for v in hsv_max)]),
            ])
        )

        super()._on_input_lines_updated()

    def _on_button_triggered(self, sender: Component) -> None:
        if sender.get_name() == "b-save":
            model: LaserDetectionModel | None = self.get_laser_detection_model()
            if model is None:
                get_app().make_toast(
                    Toast(
                        self,
                        "error",
                        "Failed to determine laser detection model",
                    )

                )
                return

            def validator(name: str) -> str | None:
                if name == "":
                    return "Please enter a file for this model"
                return None

            def already_exist_checker(name: str) -> bool:
                return repo.laser_detection.exists(name)

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
                    profile = LaserDetectionProfile(
                        name=name,
                        model=model,
                    )
                    repo.laser_detection.put(profile)
                    get_app().make_toast(
                        Toast(
                            self,
                            "info",
                            f"Saved laser detection model as {name}",
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

        if sender.get_name() == "b-clear":
            def callback(button_name: str | None) -> None:
                get_app().close_dialog()
                if button_name == "Yes":
                    self._input_lines.clear()
                    self._clear_model()

            get_app().show_dialog(
                MessageDialog(
                    is_error=True,
                    message="Are you sure you want to clear the detections?",
                    buttons=("No", "Yes"),
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
        return x, y

    def _before_add_line(self, x1: int, y1: int, x2: int, y2: int) \
            -> tuple[int, int, int, int, Any] | None:
        return x1, y1, x2, y2, None

    def key_event(self, event: KeyEvent) -> bool:
        if event.down:
            if event.key == Key.W:
                cb = self.find_component(CheckBoxComponent, "cb-show-3d-model")
                cb.set_value(not cb.get_value())
            if event.key == Key.S:
                cb = self.find_component(CheckBoxComponent, "cb-snap")
                cb.set_value(not cb.get_value())
        return super().key_event(event)
