from abc import ABC, abstractmethod

from core.tk.app import Application
from core.tk.component.button import ButtonComponent
from core.tk.component.component import Component
from core.tk.component.label import LabelComponent
from core.tk.component.spacer import SpacerComponent
from scene.my_scene import MyScene


class SelectItemDelegate(ABC):
    def item_count_per_page(self) -> int:
        return 10

    @abstractmethod
    def list_name(self) -> list[str]:
        raise NotImplementedError()

    @abstractmethod
    def execute(self, name: str) \
            -> str | None:  # returns None if success, otherwise returns error message
        raise NotImplementedError()


class SelectItemScene(MyScene):
    def __init__(self, app: "Application", delegator: SelectItemDelegate):
        super().__init__(app)
        self._delegator = delegator

        self._page = 0
        names = self._delegator.list_name()
        self._names_per_page: list[list[str]] = []
        while names:
            self._names_per_page.append(names[:delegator.item_count_per_page()])
            names = names[delegator.item_count_per_page():]
        if not self._names_per_page:
            self._names_per_page.append([])

        self._last_selected_name: str | None = None

    def load_event(self):
        self.add_component(LabelComponent, "Select Item", bold=True)
        self.add_component(SpacerComponent)
        for i in range(self._delegator.item_count_per_page()):
            self.add_component(
                ButtonComponent,
                "",
                name=f"b-item-{i}",
            )
        self.add_component(SpacerComponent)
        self.add_component(LabelComponent, "", name="l-info")
        self.add_component(SpacerComponent)
        self.add_component(ButtonComponent, "(Prev Page)", name="b-prev")
        self.add_component(ButtonComponent, "(Next Page)", name="b-next")

        self.set_page_if_possible(0)

    def update(self):
        focus_component = self.get_focus_component()
        selected_item_name = None
        for i in range(self._delegator.item_count_per_page()):
            item_btn = self.find_component(ButtonComponent, f"b-item-{i}")
            if focus_component is item_btn:
                selected_item_name = item_btn.get_text()
                break
        if self._last_selected_name != selected_item_name:
            self._last_selected_name = selected_item_name
            self.selection_change_event(self._last_selected_name)

    def selection_change_event(self, name: str | None):
        pass

    def set_page_if_possible(self, page: int):
        if page < 0:
            page = 0
        if page >= len(self._names_per_page):
            page = len(self._names_per_page) - 1
        for i in range(self._delegator.item_count_per_page()):
            self.find_component(ButtonComponent, f"b-item-{i}").set_text(
                self._names_per_page[page][i] if i < len(self._names_per_page[page]) else "-------"
            )
        self._page = page
        self.find_component(LabelComponent, "l-info").set_text(
            f"{self._page + 1}/{len(self._names_per_page)}"
        )

    def on_button_triggered(self, sender: Component) -> None:
        if isinstance(sender, ButtonComponent):
            if sender.get_name() == "b-prev":
                self.set_page_if_possible(self._page - 1)
                return
            if sender.get_name() == "b-next":
                self.set_page_if_possible(self._page + 1)
                return
            for i in range(self._delegator.item_count_per_page()):
                if sender.get_name() == f"b-item-{i}":
                    if len(self._names_per_page[self._page]) <= i:
                        return
                    name = self._names_per_page[self._page][i]
                    if not name:
                        return
                    result = self._delegator.execute(name)
                    if result is None:
                        self.get_app().make_toast(
                            "info",
                            f"Item selected: {name}"
                        )
                        self.get_app().move_back()
                    else:
                        self.get_app().make_toast(
                            "error",
                            f"Save Error: {result}"
                        )
                    return
        super().on_button_triggered(sender)
