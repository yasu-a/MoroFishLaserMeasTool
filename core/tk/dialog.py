from abc import ABC
from typing import Callable

import numpy as np

import repo.image
from core.tk.app import ApplicationWindowSize
from core.tk.component.button import ButtonComponent
from core.tk.component.component import Component
from core.tk.component.label import LabelComponent
from core.tk.component.line_edit import LineEditComponent
from core.tk.component.spacer import SpacerComponent
from core.tk.component.toast import Toast
from core.tk.event import KeyEvent
from core.tk.global_state import get_app
from core.tk.key import Key
from core.tk.rendering import UIRenderingContext
from core.tk.scene import Scene


class Dialog(Scene, ABC):
    pass


class MessageDialog(Dialog):
    def __init__(
            self,
            *,
            message: str,
            buttons: tuple[str, ...] = None,
            is_error=False,
            callback: Callable[[str | None], None],
            button_width: int = 100
    ):
        super().__init__()

        if buttons is None:  # message box, otherwise query box
            self._is_query = False
            buttons = "OK",
        else:
            self._is_query = True

        if is_error:
            self._context_name = "error"
        elif self._is_query:
            self._context_name = "query"
        else:
            self._context_name = "info"

        self._message = message
        self._buttons = buttons
        self._is_error = is_error
        self._callback = callback
        self._button_width = button_width

        self._active_button_index = 0

    def load_event(self):
        self.add_component(LabelComponent(self, self._message))
        self.add_component(SpacerComponent(self))
        self.add_component(LabelComponent(self, "", name="l-buttons"))
        super().load_event()

    def update(self):
        text = "   ".join(
            (
                f"[ {bt} ]" if i == self._active_button_index
                else f"  {bt}  "
            )
            for i, bt in enumerate(self._buttons)
        )
        self.find_component(LabelComponent, "l-buttons").set_text(text)

    def create_background(self, window_size: ApplicationWindowSize) -> np.ndarray | None:
        im = np.zeros((500, 500, 3), np.uint8)
        im = window_size.coerce(im)
        return im

    def render_ui(self, ctx: UIRenderingContext) -> UIRenderingContext:
        with ctx.enter_sub_context(self._context_name):
            ctx.canvas.fullscreen_fill(color=ctx.bg_color)
            return super().render_ui(ctx)

    def select_next_button(self):
        self._active_button_index = (self._active_button_index + 1) % len(self._buttons)

    def select_previous_button(self):
        self._active_button_index = (self._active_button_index - 1) % len(self._buttons)

    def key_event(self, event: KeyEvent) -> bool:
        if event.down:
            if event.key == Key.LEFT or event.key == Key.UP:
                self.select_previous_button()
                return True
            if event.key == Key.RIGHT or event.key == Key.DOWN:
                self.select_next_button()
                return True
            if event.key == Key.ENTER:
                self._callback(self._buttons[self._active_button_index])
                return True
            if event.key == Key.ESCAPE:
                self._callback(None)
                return True
        return super().key_event(event)


class SelectItemDialog(Dialog):
    def __init__(
            self,
            *,
            title: str,
            items: list[str],
            callback: Callable[[str | None], None],
            n_items_per_page: int = 10,
    ):
        super().__init__()

        self._title = title

        self._context_name = "query"

        self._items = items
        self._callback = callback
        self._n_items_per_page = n_items_per_page

        self._current_item_index = 0
        self._current_page = 0

        self._page_items: list[list[str]] = []
        while items:
            self._page_items.append(items[:n_items_per_page])
            items = items[n_items_per_page:]
        if not self._page_items:
            self._page_items.append([])

        self._last_item: str | None = None

    def load_event(self):
        self.add_component(LabelComponent(self, self._title, bold=True))
        self.add_component(SpacerComponent(self))
        self.add_component(LabelComponent(self, "", name="l-list"))
        self.add_component(SpacerComponent(self))
        self.add_component(LabelComponent(self, "", name="l-pages"))
        super().load_event()

    def selection_change_event(self, item: str) -> None:
        pass

    def get_current_item(self) -> str | None:
        try:
            return self._page_items[self._current_page][self._current_item_index]
        except IndexError:
            return None

    def update(self):
        cur_item = self.get_current_item()
        if self._last_item != cur_item:
            self._last_item = cur_item
            self.selection_change_event(self._last_item)

        text = "\n".join(
            (
                f" -> {item}" if i == self._current_item_index
                else f"    {item}"
            )
            for i, item in enumerate(self._page_items[self._current_page])
        )
        self.find_component(LabelComponent, "l-list").set_text(text)

        text = f"Page {self._current_page + 1}/{len(self._page_items)}"
        self.find_component(LabelComponent, "l-pages").set_text(text)

    def create_background(self, window_size: ApplicationWindowSize) -> np.ndarray | None:
        im = np.zeros((500, 500, 3), np.uint8)
        im = window_size.coerce(im)
        return im

    def render_ui(self, ctx: UIRenderingContext) -> UIRenderingContext:
        with ctx.enter_sub_context(self._context_name):
            ctx.canvas.fullscreen_fill(color=ctx.bg_color)
            return super().render_ui(ctx)

    def _try_set_index(self, page=None, item_index=None, page_delta=None, item_index_delta=None):
        new_page = self._current_page if page is None else page
        new_item_index = self._current_item_index if item_index is None else item_index

        if page_delta is not None:
            new_page += page_delta
        if item_index_delta is not None:
            new_item_index += item_index_delta

        if new_page < 0:
            new_page = 0
        if new_page >= len(self._page_items):
            new_page = len(self._page_items) - 1
        new_item_index %= len(self._page_items[new_page])

        self._current_page, self._current_item_index = new_page, new_item_index

    def select_next_page(self):
        self._try_set_index(
            page_delta=1,
            item_index_delta=0,
        )

    def select_previous_page(self):
        self._try_set_index(
            page_delta=-1,
            item_index_delta=0,
        )

    def select_next_item(self):
        self._try_set_index(
            page_delta=0,
            item_index_delta=1,
        )

    def select_previous_item(self):
        self._try_set_index(
            page_delta=0,
            item_index_delta=-1,
        )

    def key_event(self, event: KeyEvent) -> bool:
        if event.down:
            if event.key == Key.LEFT:
                self.select_previous_page()
                return True
            if event.key == Key.RIGHT:
                self.select_next_page()
                return True
            if event.key == Key.UP:
                self.select_previous_item()
                return True
            if event.key == Key.DOWN:
                self.select_next_item()
                return True
            if event.key == Key.ENTER:
                self._callback(self.get_current_item())
                return True
            if event.key == Key.ESCAPE:
                self._callback(None)
                return True
        return super().key_event(event)


class SelectImageItemDialog(SelectItemDialog):
    def selection_change_event(self, name: str | None):
        super().selection_change_event(name)
        if name is not None:
            scene = get_app().get_active_scene()
            if scene is not None:
                scene.set_picture_in_picture(repo.image.get(name).data)

    def unload_event(self):
        super().unload_event()
        scene = get_app().get_active_scene()
        if scene is not None:
            scene.set_picture_in_picture(None)


class InputNameDialog(Dialog):
    def __init__(
            self,
            *,
            title: str,
            validator: Callable[[str], str | None] = None,  # input -> None if valid else error msg
            already_exist_checker: Callable[[str], bool] = None,
            callback: Callable[[str | None], None],
    ):
        super().__init__()

        self._context_name = "query"

        self._title = title
        self._validator = validator or (lambda name: None)
        self._already_exist_checker = already_exist_checker or (lambda name: False)
        self._callback = callback

    def load_event(self):
        self.add_component(LabelComponent(self, self._title, bold=True))
        self.add_component(SpacerComponent(self))
        self.add_component(LineEditComponent(self, name="e-name"))
        self.add_component(LabelComponent(self, "", name="l-info"))
        self.add_component(SpacerComponent(self))
        self.add_component(ButtonComponent(self, "OK", name="b-ok"))
        self.add_component(ButtonComponent(self, "CANCEL", name="b-cancel"))
        super().load_event()

    def run_validation(self):
        name = self.find_component(LineEditComponent, "e-name").get_value()

        info_text = " "
        ok_button_text = "OK"

        already_exist_check_result = self._already_exist_checker(name)
        if already_exist_check_result:
            info_text = f"Name '{name}' already exists."
            ok_button_text = "OVERWRITE"

        validation_result = self._validator(name)
        if validation_result is not None:
            info_text = validation_result

        self.find_component(ButtonComponent, "b-ok").set_text(ok_button_text)
        self.find_component(LabelComponent, "l-info").set_text(info_text)

    def start_event(self):
        self.run_validation()
        super().start_event()

    def _on_value_changed(self, sender: "Component") -> None:
        if sender.get_name() == "e-name":
            self.run_validation()
            return
        super()._on_button_triggered(sender)

    def _on_button_triggered(self, sender: "Component") -> None:
        if sender.get_name() == "b-ok":
            name = self.find_component(LineEditComponent, "e-name").get_value()
            validation_result = self._validator(name)
            if validation_result is not None:
                get_app().make_toast(
                    Toast(
                        self,
                        "error",
                        validation_result,
                    )
                )
                return
            self._callback(name)
            return
        if sender.get_name() == "b-cancel":
            self._callback(None)
            return

        super()._on_button_triggered(sender)

    def create_background(self, window_size: ApplicationWindowSize) -> np.ndarray | None:
        im = np.zeros((500, 500, 3), np.uint8)
        im = window_size.coerce(im)
        return im

    def render_ui(self, ctx: UIRenderingContext) -> UIRenderingContext:
        with ctx.enter_sub_context(self._context_name):
            ctx.canvas.fullscreen_fill(color=ctx.bg_color)
            return super().render_ui(ctx)
