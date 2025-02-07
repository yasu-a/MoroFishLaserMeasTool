import cv2

from app_tk.app import Application
from app_tk.component.component import Component
from app_tk.event import KeyEvent
from app_tk.key import Key
from app_tk.rendering import RenderingContext, RenderingResult
from app_tk.scene import Scene


class LineEditComponent(Component):
    def __init__(
            self,
            app: "Application",
            scene: "Scene",
            value: str = "",
            max_length: int = 25,
            name: str = None,
    ):
        super().__init__(app, scene, name=name)
        self._max_length = max_length
        self._value = value

    def get_value(self):
        return self._value

    def set_value(self, value: str) -> None:
        self._value = value[:self._max_length]
        self._scene.on_value_changed(self)

    def render(self, ctx: RenderingContext) -> RenderingResult:
        cv2.rectangle(
            ctx.canvas,
            (ctx.left + 20, ctx.top),
            (ctx.left + 300, ctx.top + ctx.font_height),
            ctx.color,
            1,
        )
        cv2.putText(
            ctx.canvas,
            self._value,
            (ctx.left + 30, ctx.top + ctx.font_offset_y),
            ctx.font,
            ctx.scale,
            ctx.color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        if self._scene.get_focus_component() is self:
            cv2.putText(
                ctx.canvas,
                ">",
                (ctx.left, ctx.top + ctx.font_offset_y),
                ctx.font,
                ctx.scale,
                ctx.color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )
        return RenderingResult(
            height=ctx.font_height,
        )

    def key_event(self, event: KeyEvent) -> bool:
        if self._scene.get_focus_component() is self:
            if event.down:
                mapping = Key.printable_char_map()
                char = mapping.get((event.key, event.modifiers))
                prev_value = self._value
                if char is not None:
                    self._value += char
                    if len(self._value) > self._max_length:
                        self._value = self._value[:self._max_length]
                    if prev_value != self._value:
                        self._scene.on_value_changed(self)
                    return True
                elif event.key == Key.BACKSPACE:
                    if len(self._value) > 0:
                        self._value = self._value[:-1]
                    if prev_value != self._value:
                        self._scene.on_value_changed(self)
                    return True
        return super().key_event(event)

    def focus_count(self) -> int:
        return 1
