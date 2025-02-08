from typing import cast

import repo.distortion
from core.tk.component.button import ButtonComponent
from core.tk.component.component import Component
from core.tk.component.global_state import get_app
from core.tk.component.label import LabelComponent
from core.tk.component.spacer import SpacerComponent
from my_app import MyApplication
from scene.my_scene import MyScene
from scene.select_item import SelectItemDelegate, SelectItemScene


class DistortionSelectItemDelegate(SelectItemDelegate):
    def list_name(self) -> list[str]:
        return repo.distortion.list_names()

    def execute(self, name: str) -> str | None:
        cast(MyApplication, get_app()).active_profile_names.distortion_profile_name = name
        return None


class SelectProfileMenuScene(MyScene):
    def load_event(self):
        self.add_component(LabelComponent(self, "Profile Config", bold=True))
        self.add_component(SpacerComponent(self))
        self.add_component(ButtonComponent(self, "Distortion Profile", name="b-distortion"))
        self.add_component(LabelComponent(self, "", name="l-distortion"))
        self.add_component(SpacerComponent(self))
        self.add_component(ButtonComponent(self, "Camera Parameter Profile", name="b-camera-param"))
        self.add_component(LabelComponent(self, "", name="l-camera-param"))
        self.add_component(SpacerComponent(self))
        self.add_component(ButtonComponent(self, "Laser Parameter Profile", name="b-laser-param"))
        self.add_component(LabelComponent(self, "", name="l-laser-param"))
        self.add_component(SpacerComponent(self))
        self.add_component(ButtonComponent(self, "Back", name="b-back"))

    def show_event(self):
        active_profile_names = cast(MyApplication, get_app()).active_profile_names

        if active_profile_names.distortion_profile_name is None:
            self.find_component(LabelComponent, "l-distortion").set_text(
                "(NO PROFILE SELECTED)",
            )
        else:
            self.find_component(LabelComponent, "l-distortion").set_text(
                active_profile_names.distortion_profile_name,
            )

        if active_profile_names.camera_profile_name is None:
            self.find_component(LabelComponent, "l-camera-param").set_text(
                "(NO PROFILE SELECTED)",
            )
        else:
            self.find_component(LabelComponent, "l-camera-param").set_text(
                active_profile_names.camera_profile_name,
            )

        if active_profile_names.laser_profile_name is None:
            self.find_component(LabelComponent, "l-laser-param").set_text(
                "(NO PROFILE SELECTED)",
            )
        else:
            self.find_component(LabelComponent, "l-laser-param").set_text(
                active_profile_names.laser_profile_name,
            )

    def _on_button_triggered(self, sender: Component) -> None:
        if isinstance(sender, ButtonComponent):
            if sender.get_name() == "b-distortion":
                get_app().move_to(
                    SelectItemScene(
                        DistortionSelectItemDelegate(),
                    )
                )
                return
            if sender.get_name() == "b-camera-param":
                return
            if sender.get_name() == "b-laser-param":
                return
            if sender.get_name() == "b-back":
                get_app().move_back()
                return
