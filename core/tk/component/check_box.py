import cv2

from core.tk.app import Application
from core.tk.component.component import Component
from core.tk.event import KeyEvent
from core.tk.key import Key
from core.tk.rendering import RenderingContext, RenderingResult
from core.tk.scene import Scene


class CheckBoxComponent(Component):
    def __init__(self, app: "Application", scene: "Scene", text: str, value: bool = False,
                 name: str = None):
        super().__init__(app, scene, name=name)
        self._text = text.replace("\n", " ")
        self._value = value

    def get_text(self) -> str:
        return self._text

    def set_text(self, text: str) -> None:
        self._text = text.replace("\n", " ")

    def get_value(self) -> bool:
        return self._value

    def set_value(self, value: bool) -> None:
        self._value = value
        self._scene.on_value_changed(self)

    def render(self, ctx: RenderingContext) -> RenderingResult:
        x = ctx.left
        y = ctx.top + ctx.font_offset_y
        cv2.putText(
            ctx.canvas,
            "[",
            (x + 20, y),
            ctx.font,
            ctx.scale,
            ctx.color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            ctx.canvas,
            "X" if self._value else "",
            (x + 30, y),
            ctx.font,
            ctx.scale,
            ctx.color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            ctx.canvas,
            "]",
            (x + 40, y),
            ctx.font,
            ctx.scale,
            ctx.color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            ctx.canvas,
            self._text,
            (x + 50, y),
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
                (x, y),
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
        if event.down:
            if self._scene.get_focus_component() is self:
                if event.key == Key.SPACE or event.key == Key.ENTER \
                        or event.key == Key.LEFT or event.key == Key.RIGHT:
                    self._value = not self._value
                    self._scene.on_value_changed(self)
                    return True
        return super().key_event(event)

    def focus_count(self) -> int:
        return 1
