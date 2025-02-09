import repo.distortion
import repo.global_config
from core.tk.component.button import ButtonComponent
from core.tk.component.component import Component
from core.tk.component.label import LabelComponent
from core.tk.component.spacer import SpacerComponent
from core.tk.dialog import SelectItemDialog
from core.tk.global_state import get_app
from scene.my_scene import MyScene


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

    def start_event(self):
        active_profile_names = repo.global_config.get().active_profile_names

        if active_profile_names.distortion_profile_name is None:
            self.find_component(LabelComponent, "l-distortion").set_text(
                "(NO PROFILE SELECTED)",
            )
        else:
            self.find_component(LabelComponent, "l-distortion").set_text(
                active_profile_names.distortion_profile_name,
            )

        if active_profile_names.camera_param_profile_name is None:
            self.find_component(LabelComponent, "l-camera-param").set_text(
                "(NO PROFILE SELECTED)",
            )
        else:
            self.find_component(LabelComponent, "l-camera-param").set_text(
                active_profile_names.camera_param_profile_name,
            )

        if active_profile_names.laser_param_profile_name is None:
            self.find_component(LabelComponent, "l-laser-param").set_text(
                "(NO PROFILE SELECTED)",
            )
        else:
            self.find_component(LabelComponent, "l-laser-param").set_text(
                active_profile_names.laser_param_profile_name,
            )

    def _on_button_triggered(self, sender: Component) -> None:
        if isinstance(sender, ButtonComponent):
            if sender.get_name() == "b-distortion":
                def callback(item: str | None):
                    global_config = repo.global_config.get()
                    global_config.active_profile_names.distortion_profile_name = item
                    repo.global_config.put(global_config)
                    get_app().close_dialog()

                get_app().show_dialog(
                    SelectItemDialog(
                        items=repo.distortion.list_names(),
                        title="Select Distortion Profile",
                        callback=callback,
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
