import re

import repo.global_config
from core.tk.component.button import ButtonComponent
from core.tk.component.component import Component
from core.tk.component.global_state import get_app
from core.tk.component.label import LabelComponent
from core.tk.component.spacer import SpacerComponent
from core.tk.component.spin_box import SpinBoxComponent
from core.tk.component.toast import Toast
from model.camera_spec import CameraSpec
from model.global_config import GlobalConfig
from scene.my_scene import MyScene
from scene.select_item import SelectItemDelegate, SelectItemScene


class CameraResolutionSelectItemDelegate(SelectItemDelegate):
    # noinspection SpellCheckingInspection
    RESOLUTIONS = [
        ("144p", 256, 144),
        ("240p", 427, 240),
        ("360p", 640, 360),
        ("480p SD", 720, 480),
        ("720p HD", 1280, 720),
        ("HD", 1440, 1080),
        ("1080p 2K/FHD", 1920, 1080),
        ("QCIF", 176, 144),
        ("QVGA", 320, 240),
        ("WQVGA", 400, 240),
        ("WQVGA", 480, 272),
        ("HVGA", 480, 320),
        ("VGA", 640, 480),
        ("WVGA", 800, 480),
        ("SVGA", 800, 600),
        ("WSVGA", 1024, 600),
        ("XGA", 1024, 768),
        ("WXGA", 1280, 768),
        ("XGA+", 1152, 864),
        ("WXGA", 1280, 800),
        ("FWXGA", 1366, 768),
        ("Quad-VGA", 1280, 960),
        ("WXGA+", 1440, 900),
        ("SXGA", 1280, 1024),
        ("SXGA+", 1400, 1050),
        ("WSXGA", 1600, 1024),
        ("WSXGA+", 1680, 1050),
        ("UXGA", 1600, 1200),
        ("WUXGA", 1920, 1200),
        ("QWXGA", 2048, 1152),
        ("QXGA", 2048, 1536),
        ("WQHD", 2560, 1440),
        ("WQXGA", 2560, 1600),
        ("QWXGA+", 2880, 1800),
        ("WQHD+", 3200, 1800),
        ("QUXGA", 3200, 2400),
        ("4K/UHD", 3840, 2160),
        ("QUXGA Wide", 3840, 2400),
        ("4K Digital Cinema", 4096, 2160),
        ("8K/SHV", 7680, 4320),
    ]
    RESOLUTIONS = sorted(RESOLUTIONS, key=lambda x: x[1] * x[2])

    def item_count_per_page(self) -> int:
        return 20

    @classmethod
    def _resolution_to_text(cls, res: tuple[str, int, int]):  # res_name, width, height
        res_name, width, height = res
        return f"{width:>4d} x {height:>4d} ({res_name})"

    @classmethod
    def _text_to_resolution(cls, text: str) -> tuple[int, int]:  # width, height
        m = re.search(r"(\d+)\s+x\s+(\d+)\s\(", text)
        assert m, text
        width, height = int(m.group(1)), int(m.group(2))
        return width, height

    def list_name(self) -> list[str]:
        return [
            self._resolution_to_text(res)
            for res in self.RESOLUTIONS
        ]

    def execute(self, name: str) -> str | None:
        global_config = repo.global_config.get()
        width, height = self._text_to_resolution(name)
        global_config.camera_spec = CameraSpec(
            width=width,
            height=height,
            fps=30,
        )
        repo.global_config.put(global_config)
        return None


class GlobalConfigScene(MyScene):
    def load_event(self):
        self.add_component(LabelComponent(self, "Global Configuration", bold=True))
        self.add_component(LabelComponent(self, "[Camera]"))
        self.add_component(LabelComponent(self, "Camera ID"))
        self.add_component(
            SpinBoxComponent(
                self,
                min_value=0,
                max_value=10,
                name="sb-camera-id",
            )
        )
        self.add_component(LabelComponent(self, "Camera Resolution"))
        self.add_component(ButtonComponent(self, "", name="b-camera-resolution"))
        self.add_component(LabelComponent(self, "Camera FPS"))
        self.add_component(
            SpinBoxComponent(
                self,
                min_value=1,
                max_value=300,
                name="sb-camera-fps",
            )
        )
        self.add_component(SpacerComponent(self))
        self.add_component(ButtonComponent(self, "Back", name="b-back"))

    def show_event(self):
        global_config: GlobalConfig = repo.global_config.get()
        self.find_component(SpinBoxComponent, "sb-camera-id").set_value(
            global_config.camera_dev_id,
        )
        self.find_component(ButtonComponent, "b-camera-resolution").set_text(
            f"{global_config.camera_spec.width} x {global_config.camera_spec.height}"
        )
        self.find_component(SpinBoxComponent, "sb-camera-fps").set_value(
            int(global_config.camera_spec.fps),
        )

    def unload_event(self):
        global_config: GlobalConfig = repo.global_config.get()
        camera_dev_id = self.find_component(SpinBoxComponent, "sb-camera-id").get_value()
        global_config.camera_dev_id = camera_dev_id
        global_config.camera_spec = CameraSpec(
            width=global_config.camera_spec.width,
            height=global_config.camera_spec.height,
            fps=self.find_component(SpinBoxComponent, "sb-camera-fps").get_value(),
        )
        repo.global_config.put(global_config)
        get_app().make_toast(
            Toast(
                self,
                "info",
                "Configuration updated"
            )
        )
        get_app().move_back()

    def _on_button_triggered(self, sender: Component) -> None:
        if sender.get_name() == "b-camera-resolution":
            get_app().move_to(
                SelectItemScene(
                    CameraResolutionSelectItemDelegate(),
                )
            )
            return
        if sender.get_name() == "b-back":
            get_app().move_back()
